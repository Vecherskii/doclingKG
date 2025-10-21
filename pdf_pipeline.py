# pdf_pipeline.py
# End-to-end on-prem pipeline (Windows-friendly, GPU-capable)
# - Docling: text + formula enrichment + referenced images
# - LaTeX tidy (light, safe fixes)
# - Image captions via Qwen2-VL (GPU if available)
# - Final outputs: Markdown (.md), Plain text (.txt), and HTML (.html)

import os
import re
import sys
import subprocess
from typing import Optional

_MATH_INLINE = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', flags=re.DOTALL)
_MATH_BLOCK  = re.compile(r'\$\$(.+?)\$\$', flags=re.DOTALL)

def _split_display_math_from_prose(tex_block: str) -> tuple[str, str]:
    """
    Heuristic: if a display-math block contains prose-y lines, peel them out.
    Returns (clean_math, trailing_prose).
    A line is considered prose-y if it has many letters/spaces and very few math tokens.
    """
    lines = tex_block.splitlines()
    math_lines, prose_lines = [], []
    for ln in lines:
        # crude features
        letters = sum(c.isalpha() for c in ln)
        backslashes = ln.count("\\")
        digits = sum(c.isdigit() for c in ln)
        # prose if many letters and few math markers
        if letters >= 0.5 * max(1, len(ln)) and backslashes <= 1 and digits <= 2:
            prose_lines.append(ln.strip())
        else:
            math_lines.append(ln)
    math_clean = "\n".join(math_lines).strip()
    prose_tail = (" " + " ".join(prose_lines).strip()) if prose_lines else ""
    return math_clean, prose_tail

def _tidy_math_fragment(tex: str) -> str:
    s = tex

    # 0) Normalize common Unicode math symbols to LaTeX macros
    #    ℓ -> \ell
    s = s.replace("ℓ", r"\ell")

    # 1) General spacing tighten
    s = re.sub(r'\\text\s*\{\s*', r'\\text{', s)  # \text {..} -> \text{..}
    s = re.sub(r'_\s*\{', r'_{', s)               # _ { -> _{
    s = re.sub(r'\^\s*\{', r'^{', s)              # ^ { -> ^{

    # \left( \right) and friends
    s = re.sub(r'\\left\s*\(', r'\\left(', s); s = re.sub(r'\\right\s*\)', r'\\right)', s)
    s = re.sub(r'\\left\s*\[', r'\\left[', s);   s = re.sub(r'\\right\s*\]', r'\\right]', s)
    s = re.sub(r'\\left\s*\{', r'\\left{', s);   s = re.sub(r'\\right\s*\}', r'\\right}', s)

    # function-like: \ell ( x ) -> \ell(x), \max ( ... ) -> \max(...)
    s = re.sub(r'(\\[A-Za-z]+)\s*\(\s*', r'\1(', s)
    s = re.sub(r'\s*\)\s*', r')', s)

    # brackets/commas
    s = re.sub(r'\[\s*', r'[', s); s = re.sub(r'\s*\]', r']', s)
    s = re.sub(r'\{\s*', r'{', s); s = re.sub(r'\s*\}', r'}', s)
    s = re.sub(r'\s*,\s*', r', ', s)

    # 2) Normalize "VarWin B" / "VarEnd B" with optional spaces/underscore
    s = re.sub(r'\bVarWin\s*_?\s*B\b', r'\\text{VarWin}_{B}', s)
    s = re.sub(r'\bVarEnd\s*_?\s*B\b', r'\\text{VarEnd}_{B}', s)

    # Ensure [\ell] written with macro if Unicode slipped in earlier
    s = s.replace('[ ℓ ]', r'[\ell]').replace('[ℓ]', r'[\ell]')

    # 3) Fix split decimals INSIDE math (order matters)
    # a) three separated digits: 6 7 5 -> 6.75
    s = re.sub(r'(\d)\s(\d)\s(\d)\b', r'\1.\2\3', s)
    # b) one + two digits: 6 75 -> 6.75
    s = re.sub(r'(\d)\s(\d{2})\b', r'\1.\2', s)
    # c) one + one digit at end: 3 1 -> 3.1 (only if not already fixed)
    s = re.sub(r'(\d)\s(\d)\b', r'\1.\2', s)
    # d) collapse accidental "6.7.5" -> "6.75"
    s = re.sub(r'(\d)\.(\d)\.(\d)\b', r'\1.\2\3', s)

    return s

# --------- Config defaults ---------
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"   # gated: set HF_TOKEN env
DEFAULT_MAX_NEW_TOKENS = 200
MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# --------- Utilities ---------
def _log(msg: str): print(msg, flush=True)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def _detect_device(requested: Optional[str] = None) -> str:
    # Respect explicit user choice first
    if requested in {"cuda", "cpu"}:
        return requested
    # Auto-detect
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# --------- Step A: run Docling CLI ---------
def run_docling(pdf_path: str, out_dir: str, enrich_formula: bool = True) -> str:
    """
    Uses the Docling CLI to produce:
      - Markdown with referenced images
    Returns the path to the MD.
    """
    _ensure_dir(out_dir)
    cmd = [
        "docling",
        "--to", "md",
        "--image-export-mode", "referenced",
        "--output", out_dir
    ]
    if enrich_formula:
        cmd.insert(1, "--enrich-formula")
    cmd.append(pdf_path)
    _log(f"[INFO] Running Docling: {' '.join(cmd)}")
    # Let Docling print its own logs
    subprocess.run(cmd, check=True)
    # Resolve the output filename (Docling keeps input basename)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(out_dir, f"{base}.md")
    if not os.path.exists(md_path):
        # Fallback: list .md in out_dir
        cands = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".md")]
        if not cands:
            raise FileNotFoundError("Docling produced no .md output.")
        md_path = max(cands, key=os.path.getmtime)
    _log(f"[INFO] Docling MD: {md_path}")
    return md_path

# --------- Step B: LaTeX tidy (safe, minimal) ---------
_LATEX_FIXES = [
    (r'\\intertext\s*\{[^}]*\}', ''),            # remove stray \intertext{...}
    (r'\\text\s*\{', r'\\text{'),                # tighten \text { -> \text{
    (r'_\s*\{', r'_{'),                          # tighten spaces before subscripts
    (r'\^\s*\{', r'^{'),                         # tighten spaces before superscripts
    (r'\\left\s*\(', r'\\left('),                # tighten \left (
    (r'\\right\s*\)', r'\\right)'),              # tighten \right )
    # Optional: fix common split decimals like "6 75" -> "6.75" ONLY when obvious pattern
    (r'(?<=\b6)\s(?=75\b)', '.'),               # very surgical for 6.75 seen in your sample
]

def tidy_latex_in_markdown(md_path: str) -> str:
    text = open(md_path, encoding="utf-8").read()

    # remove stray \intertext{...}
    text = re.sub(r'\\intertext\s*\{[^}]*\}', '', text)

    # $$...$$ blocks: tidy + peel prose out of the block
    def _fix_block(m: re.Match) -> str:
        block = m.group(1)
        block_tidy = _tidy_math_fragment(block)
        math_only, prose_tail = _split_display_math_from_prose(block_tidy)
        # Put prose AFTER the display math
        return "$$\n" + math_only + "\n$$" + prose_tail
    text = _MATH_BLOCK.sub(_fix_block, text)

    # $...$ inline: tidy only
    def _fix_inline(m: re.Match) -> str:
        return "$" + _tidy_math_fragment(m.group(1)) + "$"
    text = _MATH_INLINE.sub(_fix_inline, text)

    # conservative prose decimal fix: "1 . 5%" -> "1.5%"
    text = re.sub(r'(\d)\s*\.\s*(\d)(?=%)', r'\1.\2', text)

    # light global cleanup
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    out = os.path.join(os.path.dirname(md_path),
                       os.path.splitext(os.path.basename(md_path))[0] + "_latex_clean.md")
    open(out, "w", encoding="utf-8").write(text)
    _log(f"[DONE] LaTeX tidy -> {out}")
    return out



# --------- Step C: image -> caption replacement (Qwen2-VL) ---------
_IMAGE_MD_PATTERN = re.compile(r'!\[[^\]]*\]\(\s*([^) \t]+)(?:\s+"[^"]*")?\s*\)')

def _resolve_path(md_path: str, raw: str) -> str:
    raw = raw.strip().strip("'\"")
    if raw.startswith("http://") or raw.startswith("https://"): return raw
    if (":" in raw) or raw.startswith("\\\\"): return os.path.normpath(raw)
    return os.path.normpath(os.path.join(os.path.dirname(md_path), raw))

def _load_qwen(model_id: str, device: str):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as ModelCls

    kw = {}
    tok = os.getenv("HF_TOKEN")
    if tok: kw["token"] = tok

    _log(f"[INFO] Loading model on {device}: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False, **kw)

    # If accelerate is present, device_map='auto' will shard automatically
    try:
        model = ModelCls.from_pretrained(model_id, device_map="auto", dtype="auto", **kw)
    except Exception:
        model = ModelCls.from_pretrained(model_id, torch_dtype="auto", **kw)
        try:
            import torch
            model = model.to(device)
        except Exception:
            pass

    # modest precision speedup if torch is available
    try:
        import torch
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return processor, model

def _qwen_caption(img_path: str, processor, model, max_new_tokens: int, context: str = "") -> str:
    import torch
    from PIL import Image
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()  # silence "temperature/top_p/top_k" warnings

    with Image.open(img_path) as im:
        im = im.convert("RGB")

        # Tighter, factual prompt
        base_prompt = (
            "You are a scientific figure analyst for math/ML papers.\n"
            "Describe the figure in 1–2 precise sentences.\n"
            "- If axes/labels/units are visible, name them.\n"
            "- Mention trend direction and any thresholds/lines with values ONLY if clearly legible.\n"
            "- If not legible, say 'axes unlabeled'.\n"
            "- Do NOT guess variables or numbers.\n"
        )
        final_prompt = base_prompt + (f"\nContext:\n{context.strip()}" if context.strip() else "")

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_prompt}]}]
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=chat_text, images=[im], return_tensors="pt")

        # move all tensors to the model's device
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(model_device)

        # Greedy, deterministic (no unsupported args)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
        input_len = inputs["input_ids"].shape[1]
        tail = gen[:, input_len:]
        caption = processor.batch_decode(tail, skip_special_tokens=True)[0].strip()

        # strip any stray role label
        lc = caption.lower()
        if lc.startswith(("system", "user", "assistant")):
            caption = caption.split("\n", 1)[-1].strip()
        return caption

def replace_images_with_captions(md_path: str, model_id: str, device: str, max_new_tokens: int) -> str:
    txt = open(md_path, encoding="utf-8").read()

    if not _IMAGE_MD_PATTERN.search(txt):
        _log("[INFO] No images detected in MD; skipping caption step.")
        return md_path

    processor, model = _load_qwen(model_id, device)

    # Build a search buffer for context extraction per image
    def _sub(m: re.Match) -> str:
        raw = m.group(1)
        raw = raw.split(' "')[0]  # strip optional title
        resolved = _resolve_path(md_path, raw)
        _log(f"[INFO] -> {resolved}")

        # Context: look at up to 2 non-empty lines before the image; prefer 'Figure...'
        start_idx = m.start()
        prefix = txt[:start_idx]
        lines = [ln.strip() for ln in prefix.splitlines() if ln.strip()]
        ctx = ""
        if lines:
            ctx_lines = lines[-2:]
            for ln in reversed(ctx_lines):
                if ln.lower().startswith(("figure", "fig.")):
                    ctx_lines = [ln]; break
            ctx = "\n".join(ctx_lines)

        if resolved.startswith(("http://", "https://")):
            _log(f"[WARN] Remote image skipped: {resolved}")
            return f"[Image omitted: {os.path.basename(resolved)}]"

        if not os.path.exists(resolved):
            _log(f"[WARN] Missing file: {resolved}")
            return f"[Image omitted: {os.path.basename(resolved)}]"

        try:
            desc = _qwen_caption(resolved, processor, model, max_new_tokens, context=ctx)
            _log(f"[OK] Captioned: {os.path.basename(resolved)}")
            return f"[Image: {desc}]"
        except Exception as e:
            _log(f"[WARN] Caption failed for {resolved}: {e}")
            return f"[Image omitted: {os.path.basename(resolved)}]"

    out = _IMAGE_MD_PATTERN.sub(_sub, txt)
    outf = os.path.join(os.path.dirname(md_path), os.path.splitext(os.path.basename(md_path))[0] + "_images2text.md")
    open(outf, "w", encoding="utf-8").write(out)
    _log(f"[DONE] Image->text -> {outf}")
    return outf

# --------- Step D: write TXT + HTML ---------
def md_to_txt(md_path: str) -> str:
    """
    Very simple MD->TXT: remove the most common markdown markers but keep content,
    including [Image: ...] descriptions (that's desired).
    """
    s = open(md_path, encoding="utf-8").read()
    # strip markdown headings / bold / italics markers minimally
    s = re.sub(r'^\s{0,3}#{1,6}\s*', '', s, flags=re.MULTILINE)     # leading #'s
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)                         # **bold**
    s = re.sub(r'\*([^*]+)\*', r'\1', s)                             # *italics*
    s = re.sub(r'`{1,3}([^`]+)`{1,3}', r'\1', s)                     # inline code
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', s)                 # links -> text
    s = re.sub(r'\n{3,}', '\n\n', s)                                 # collapse empty lines
    out = os.path.join(os.path.dirname(md_path), os.path.splitext(os.path.basename(md_path))[0] + ".txt")
    open(out, "w", encoding="utf-8").write(s)
    _log(f"[DONE] TXT -> {out}")
    return out

def md_to_html(md_path: str) -> str:
    """
    Render Markdown to HTML; add MathJax for LaTeX. Falls back to <pre> if 'markdown' pkg isn't installed.
    """
    html_body = None
    s = open(md_path, encoding="utf-8").read()
    try:
        import markdown
        html_body = markdown.markdown(s, extensions=["fenced_code", "tables", "toc"])
    except Exception:
        html_body = "<pre>" + s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>"

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{os.path.basename(md_path)}</title>
<script src="{MATHJAX_CDN}" id="MathJax-script" async></script>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:900px;margin:40px auto;line-height:1.5;padding:0 16px}}
pre,code{{font-family:Consolas,Menlo,monospace}}
img{{max-width:100%}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    out = os.path.join(os.path.dirname(md_path), os.path.splitext(os.path.basename(md_path))[0] + ".html")
    open(out, "w", encoding="utf-8").write(html)
    _log(f"[DONE] HTML -> {out}")
    return out

def md_to_html_pretty(md_path: str) -> str:
    """
    Rich HTML for fast QA:
    - sticky sidebar TOC
    - readable typography
    - highlighted [Image: ...] callouts
    - MathJax for LaTeX
    """
    s = open(md_path, encoding="utf-8").read()
    try:
        import markdown
        html_body = markdown.markdown(s, extensions=["fenced_code", "tables", "toc"])
    except Exception:
        html_body = "<pre>" + s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") + "</pre>"

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{os.path.basename(md_path)} — pretty</title>
<script src="{MATHJAX_CDN}" id="MathJax-script" async></script>
<style>
:root {{
  --bg:#0b0d10; --panel:#151a21; --text:#e9eef5; --muted:#9fb0c6; --accent:#4db6ff; --callout:#1e2530; --ok:#40c463
}}
*{{box-sizing:border-box}}
body{{margin:0;display:flex;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}}
nav{{position:sticky;top:0;align-self:flex-start;width:280px;max-height:100vh;overflow:auto;background:var(--panel);border-right:1px solid #202733;padding:20px}}
nav h2{{font-size:14px;margin:0 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}}
nav ul{{list-style:none;padding:0;margin:0}}
nav li{{margin:6px 0}}
nav a{{color:var(--muted);text-decoration:none}}
nav a:hover{{color:var(--accent)}}
main{{flex:1;max-width:1000px;margin:0 auto;padding:32px}}
main h1,main h2,main h3{{scroll-margin-top:16px}}
pre,code{{font-family:Consolas,Menlo,monospace}}
blockquote{{margin:0;padding:12px 16px;background:#12171e;border-left:3px solid var(--accent)}}
.callout{{background:var(--callout);border:1px solid #243040;border-radius:12px;padding:14px 16px;margin:12px 0}}
.callout .tag{{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;background:#0f141b;color:var(--ok);border:1px solid #2b3a49;margin-right:8px}}
hr{{border:none;border-top:1px solid #223041;margin:24px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #2a3545;padding:8px}}
a{{color:var(--accent)}}
</style>
</head>
<body>
<nav>
  <h2>Contents</h2>
  <div class="toc">{'{%s}' % ''}</div>
  <!-- markdown's 'toc' extension injects a <div class="toc"> automatically;
       we include a placeholder to keep layout stable if extension missing. -->
</nav>
<main>
{html_body.replace('[Image:', '<div class="callout"><span class="tag">Image</span>').replace(']', '</div>')}
</main>
</body>
</html>"""
    out = os.path.join(os.path.dirname(md_path), os.path.splitext(os.path.basename(md_path))[0] + "_pretty.html")
    open(out, "w", encoding="utf-8").write(html)
    _log(f"[DONE] Pretty HTML -> {out}")
    return out

# --------- Orchestrator ---------
def run_pipeline(pdf_path: str,
                 out_dir: str,
                 use_gpu: Optional[bool] = None,
                 hf_token: Optional[str] = None,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 model_id: str = DEFAULT_MODEL_ID):
    """
    Full run:
      1) Docling -> MD (with formula enrichment + referenced images)
      2) Tidy LaTeX in MD
      3) Replace images with captions (Qwen2-VL, GPU if available)
      4) Produce TXT and HTML from the final MD
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    device = _detect_device("cuda" if use_gpu else None)
    _ensure_dir(out_dir)

    # 1) Docling
    md_docling = run_docling(pdf_path, out_dir, enrich_formula=True)

    # 2) LaTeX tidy
    md_clean = tidy_latex_in_markdown(md_docling)

    # 3) Image captions -> replace images with textual descriptions
    md_final = replace_images_with_captions(md_clean, model_id=model_id, device=device, max_new_tokens=max_new_tokens)

    # 4) Exports
    txt_path = md_to_txt(md_final)
    html_path = md_to_html(md_final)

    html_pretty_path = md_to_html_pretty(md_final)
    return {"md": md_final, "txt": txt_path, "html": html_path, "html_pretty": html_prety_path if 'html_prety_path' in locals() else html_pretty_path}


# --------------------------- CLI ---------------------------
def _parse_args(argv):
    """
    --pdf <path> (required)
    --out <dir> (required)
    --device <cuda|cpu> (optional; auto if omitted)
    --hf-token <token> (optional; for gated models)
    --max-new-tokens <int> (optional; default 200)
    --model-id <repo> (optional; default Qwen/Qwen2-VL-2B-Instruct)
    """
    args = {"pdf": None, "out": None, "device": None, "hf_token": None,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS, "model_id": DEFAULT_MODEL_ID}
    it = iter(range(len(argv)))
    for i in it:
        t = argv[i]
        if t == "--pdf" and i + 1 < len(argv): args["pdf"] = argv[i+1]; next(it, None)
        elif t == "--out" and i + 1 < len(argv): args["out"] = argv[i+1]; next(it, None)
        elif t == "--device" and i + 1 < len(argv): args["device"] = argv[i+1]; next(it, None)
        elif t == "--hf-token" and i + 1 < len(argv): args["hf_token"] = argv[i+1]; next(it, None)
        elif t == "--max-new-tokens" and i + 1 < len(argv):
            try: args["max_new_tokens"] = int(argv[i+1])
            except: pass
            next(it, None)
        elif t == "--model-id" and i + 1 < len(argv): args["model_id"] = argv[i+1]; next(it, None)
    return args

if __name__ == "__main__":
    a = _parse_args(sys.argv[1:])
    if not a["pdf"] or not a["out"]:
        _log("Usage: python pdf_pipeline.py --pdf <input.pdf> --out <output_dir> [--device cuda|cpu] [--hf-token <token>] [--max-new-tokens 200] [--model-id <repo>]")
        sys.exit(2)
    if a["device"] is not None and a["device"] not in {"cuda", "cpu"}:
        _log("--device must be 'cuda' or 'cpu'"); sys.exit(2)
    res = run_pipeline(
        pdf_path=a["pdf"],
        out_dir=a["out"],
        use_gpu=(a["device"] == "cuda") if a["device"] else None,
        hf_token=a["hf_token"],
        max_new_tokens=a["max_new_tokens"],
        model_id=a["model_id"]
    )
    _log(f"[RESULT] {res}")
