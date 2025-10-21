# pdf_pipeline.py — images→text captions with Qwen2-VL (chat template), Windows-friendly

import os
import re
import sys
from typing import Optional
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as _AutoModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as _AutoModelClass  # back-compat

_IMAGE_MD_PATTERN = re.compile(r'!\[[^\]]*\]\(\s*([^) \t]+)(?:\s+"[^"]*")?\s*\)')
_DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
_DEFAULT_MAX_NEW_TOKENS = 96

def _log(msg: str) -> None:
    print(msg, flush=True)

def _resolve_path(md_path: str, raw: str) -> str:
    raw = raw.strip().strip("'\"")
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if (":" in raw) or raw.startswith("\\\\"):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(os.path.dirname(md_path), raw))

def _load_model(model_id: str, hf_token: Optional[str]):
    kw = {}
    if hf_token:
        kw["token"] = hf_token
    _log(f"[INFO] Loading model: {model_id}")
    # use_fast=False avoids some tokenizer/image-processor edge cases on Windows
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False, **kw)
    try:
        model = _AutoModelClass.from_pretrained(model_id, dtype="auto", **kw)
    except TypeError:
        model = _AutoModelClass.from_pretrained(model_id, torch_dtype="auto", **kw)
    return processor, model

def _caption_image(img_path: str, processor, model, max_new_tokens: int, context: str = "") -> str:
    """
    Qwen2-VL captioning with chat template, grounded by nearby text 'context'.
    Deterministic decoding to reduce hallucinations.
    """
    from PIL import Image
    with Image.open(img_path) as im:
        im = im.convert("RGB")

        base_prompt = (
            "You are a scientific figure analyst. "
            "Write 1–2 precise sentences describing the figure for a math/ML paper. "
            "Use the provided context to stay factual. "
            "If axes or labels are unreadable, say 'axes unlabeled'. "
            "Do NOT invent variable names, thresholds, or trends not stated in the context."
        )

        # Fold in local context (e.g., 'Figure 6: ...')
        final_prompt = (base_prompt + "\n\nContext:\n" + context.strip()) if context.strip() else base_prompt

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": final_prompt}
            ]
        }]

        # Build chat text (no tokenization yet)
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Prepare both text tokens and image tensor together
        inputs = processor(text=chat_text, images=[im], return_tensors="pt")

        # Deterministic decoding (no sampling), slightly discourage repetition
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05
        )

        # Keep only generated tail (avoid prompt echo)
        input_len = inputs["input_ids"].shape[1]
        tail = gen[:, input_len:]
        caption = processor.batch_decode(tail, skip_special_tokens=True)[0].strip()

        # Small cleanup: if it still starts with a role label, drop first line
        lowers = caption.lower()
        if lowers.startswith("system") or lowers.startswith("user") or lowers.startswith("assistant"):
            caption = caption.split("\n", 1)[-1].strip()

        return caption



def replace_images_with_captions(md_path: str,
                                 out_path: Optional[str] = None,
                                 model_id: str = _DEFAULT_MODEL_ID,
                                 max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS) -> str:
    if not os.path.exists(md_path):
        raise FileNotFoundError(md_path)

    _log(f"[INFO] MD: {md_path}")
    text = open(md_path, encoding="utf-8").read()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(md_path), "input_images2text.md")

    hf_token = os.getenv("HF_TOKEN", None)
    if not _IMAGE_MD_PATTERN.search(text):
        _log("[WARN] No images found in Markdown. Writing a copy unchanged.")
        open(out_path, "w", encoding="utf-8").write(text)
        _log(f"[DONE] Wrote (unchanged): {out_path}")
        return out_path

    processor, model = _load_model(model_id, hf_token)

    def _sub(m: re.Match) -> str:
        raw = m.group(1)
        raw = raw.split(' "')[0]
        resolved = _resolve_path(md_path, raw)
        _log(f"[INFO] -> {resolved}")

        # --- harvest local context (e.g., 'Figure 6: ...') from up to ~2 preceding non-empty lines ---
        # We search back from the start of the match for the nearest lines to ground the caption
        start_idx = m.start()
        prefix = text[:start_idx]
        lines = [ln.strip() for ln in prefix.splitlines() if ln.strip()]
        ctx = ""
        if lines:
            # Take up to 2 nearest lines
            ctx_lines = lines[-2:]
            # Prefer a line that begins with 'Figure'/'Fig.' if present
            for ln in reversed(ctx_lines):
                if ln.lower().startswith("figure") or ln.lower().startswith("fig."):
                    ctx_lines = [ln]  # use only the figure caption line
                    break
            ctx = "\n".join(ctx_lines)

        if resolved.startswith("http://") or resolved.startswith("https://"):
            _log(f"[WARN] Remote image skipped: {resolved}")
            return f"[Image omitted: {os.path.basename(resolved)}]"

        if not os.path.exists(resolved):
            _log(f"[WARN] Missing file: {resolved}")
            return f"[Image omitted: {os.path.basename(resolved)}]"

        try:
            desc = _caption_image(resolved, processor, model, max_new_tokens, context=ctx)
            _log(f"[OK] Captioned: {os.path.basename(resolved)}")
            return f"[Image: {desc}]"
        except Exception as e:
            _log(f"[WARN] Caption failed for {resolved}: {e}")
            return f"[Image omitted: {os.path.basename(resolved)}]"

    new_text = _IMAGE_MD_PATTERN.sub(_sub, text)
    open(out_path, "w", encoding="utf-8").write(new_text)
    _log(f"[DONE] Wrote: {out_path}")
    return out_path

def _parse_argv(argv):
    args = {"input": None, "output": None, "model_id": _DEFAULT_MODEL_ID, "max_new_tokens": _DEFAULT_MAX_NEW_TOKENS}
    it = iter(range(len(argv)))
    for i in it:
        tok = argv[i]
        if tok == "--input" and i + 1 < len(argv):
            args["input"] = argv[i + 1]; next(it, None)
        elif tok == "--output" and i + 1 < len(argv):
            args["output"] = argv[i + 1]; next(it, None)
        elif tok == "--model-id" and i + 1 < len(argv):
            args["model_id"] = argv[i + 1]; next(it, None)
        elif tok == "--max-new-tokens" and i + 1 < len(argv):
            try: args["max_new_tokens"] = int(argv[i + 1])
            except ValueError: pass
            next(it, None)
    return args

if __name__ == "__main__":
    args = _parse_argv(sys.argv[1:])
    if not args["input"]:
        _log("Usage: python pdf_pipeline.py --input <markdown_path> [--output <out.md>] [--model-id <repo>] [--max-new-tokens 96]")
        sys.exit(2)
    replace_images_with_captions(args["input"], args["output"], args["model_id"], args["max_new_tokens"])
