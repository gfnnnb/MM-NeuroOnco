#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct evaluation script for local HuggingFace VLMs on MM-NeuroOnco closed-ended benchmark.

Features:
- CLI configurable (model/json/data/out/device/dtype)
- Option-E support (A-D / A-E auto)
- Robust image path resolution with optional extra_roots
- Optional resume by image_path
- Outputs JSONL predictions + summary JSON + config JSON
- By default does NOT save raw decoded text or absolute image paths
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
)


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a, b):
    return a / b if b else 0.0


def extract_letter(decoded: str, allow_e: bool) -> str:
    """Extract answer letter. allow_e=True => A-E, else A-D."""
    letters = "ABCDE" if allow_e else "ABCD"
    t = (decoded or "").strip().upper()
    if not t:
        return ""
    m = re.fullmatch(rf"([{letters}])", t)
    if m:
        return m.group(1)
    m = re.search(rf"(?:^|[\s\(\[\:])([{letters}])(?:[\.\)\s\:]|$)", t)
    if m:
        return m.group(1)
    m = re.findall(rf"[{letters}]", t)
    return m[-1] if m else ""


def resolve_image_path(data_root: Path, image_path_str: str, extra_roots=None):
    """
    Resolve image_path:
    1) data_root / image_path
    2) strip possible duplicated root segment
    3) try extra_roots
    """
    image_path = Path(image_path_str)

    cand = data_root / image_path
    if cand.exists():
        return cand

    root_name = data_root.name
    parts = list(image_path.parts)
    if root_name in parts:
        idx = parts.index(root_name)
        stripped = Path(*parts[idx + 1 :])
        cand2 = data_root / stripped
        if cand2.exists():
            return cand2

    if extra_roots:
        for r in extra_roots:
            r = Path(r)
            cand3 = r / image_path
            if cand3.exists():
                return cand3
            if root_name in parts:
                idx = parts.index(root_name)
                cand4 = r / Path(*parts[idx + 1 :])
                if cand4.exists():
                    return cand4

    return None


def load_vlm(model_dir: str, dtype: torch.dtype, attn_impl: str | None):
    """
    Generic VLM loader: try common model classes in order.
    """
    print(f"🔧 Loading processor from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    last_err = None
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            print(f"🔧 Trying model class: {cls.__name__}")
            kwargs = dict(
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            if attn_impl:
                kwargs["_attn_implementation"] = attn_impl
            model = cls.from_pretrained(model_dir, **kwargs)
            model.eval()
            print(f"✅ Loaded model with {cls.__name__}")
            return model, processor, cls.__name__
        except Exception as e:
            last_err = e
            print(f"⚠️ Failed with {cls.__name__}: {e}")

    raise RuntimeError(f"❌ All model classes failed. Last error: {last_err}")


def get_pad_token_id(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is not None:
            return eos_id
    return None


def build_question_text(pair: dict) -> str:
    options = pair.get("options", {}) or {}
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    allow_e = "E" in options
    valid = "A, B, C, D, or E" if allow_e else "A, B, C, or D"
    return (
        "Answer the multiple-choice question based on the MRI image.\n"
        f"Output ONLY one letter: {valid}.\n"
        "Do not output any other text.\n\n"
        f"Question: {pair['question']}\n"
        f"Options:\n{opt_text}\n"
        "Answer:"
    )


def prepare_inputs(processor, model, image: Image.Image, question_text: str, dtype: torch.dtype):
    """
    Prefer apply_chat_template. Fallback to processor(text, images).
    Return: inputs, input_len, mode
    """
    # More compatible ordering: image first then text (many VLMs prefer this)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ],
        }
    ]

    # 1) chat template tokenized
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=dtype)
        input_len = inputs["input_ids"].shape[-1]
        return inputs, input_len, "chat_template_tokenized"
    except Exception:
        pass

    # 2) prompt string from chat template
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else None
        return inputs, input_len, "chat_template_prompt"
    except Exception:
        pass

    # 3) fallback
    inputs = processor(text=question_text, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else None
    return inputs, input_len, "fallback_processor"


def load_done_keys(jsonl_path: Path):
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ip = obj.get("image_path")
                if ip:
                    done.add(ip)
            except Exception:
                continue
    return done


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Direct local VLM evaluation (Option-E supported).")

    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--json_path", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)

    p.add_argument("--extra_roots", type=str, nargs="*", default=[], help="Extra roots for resolving image paths.")

    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--cuda_visible_devices", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default=None)

    p.add_argument("--max_new_tokens", type=int, default=10)
    p.add_argument("--do_sample", action="store_true")

    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--max_qa", type=int, default=None)

    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--save_image_abs", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"✅ Set CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    json_path = Path(args.json_path)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name_for_tag = Path(args.model_dir.rstrip("/")).name.replace("/", "_")
    run_tag = args.run_tag or f"{model_name_for_tag}_direct"

    out_jsonl = out_dir / f"preds_{run_tag}.jsonl"
    out_summary = out_dir / f"summary_{run_tag}.json"
    out_missing = out_dir / f"missing_{run_tag}.txt"
    out_config = out_dir / f"config_{run_tag}.json"

    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_images is not None:
        data = data[: args.max_images]

    print(f"📦 Loaded {len(data)} samples")
    print(f"🖼️  data_root={data_root}")
    print(f"🤖 model_dir={args.model_dir}")
    print(f"💾 out_jsonl={out_jsonl}")

    model, processor, model_class = load_vlm(args.model_dir, dtype, args.attn_impl)
    pad_token_id = get_pad_token_id(processor)

    done_keys = load_done_keys(out_jsonl) if args.resume else set()
    if args.resume:
        print(f"✅ Resume enabled. Done samples: {len(done_keys)} (by image_path)")

    metrics = {
        "total": 0,
        "correct": 0,
        "binary": {"total": 0, "correct": 0},  # A-D
        "mcq": {"total": 0, "correct": 0},     # with E
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
        "img_ok": 0,
        "img_missing": 0,
        "img_open_failed": 0,
    }

    missing_images = []
    qa_count = 0

    extra_roots = [Path(x) for x in args.extra_roots]

    write_mode = "a" if args.resume else "w"
    with open(out_jsonl, write_mode, encoding="utf-8") as w:
        for item in tqdm(data, desc="Processing"):
            ip = item.get("image_path", "")
            if args.resume and ip and ip in done_keys:
                continue

            img_abs = resolve_image_path(data_root, ip, extra_roots=extra_roots)
            if img_abs is None:
                metrics["img_missing"] += 1
                missing_images.append(ip)
                continue

            try:
                image = Image.open(img_abs).convert("RGB")
                metrics["img_ok"] += 1
            except Exception as e:
                metrics["img_open_failed"] += 1
                missing_images.append(f"{ip} | open_failed: {e}")
                continue

            for qi, pair in enumerate(item.get("vqa_pairs", []), start=1):
                q_type = pair.get("question_type", "unknown")
                gt_ans = (pair.get("answer") or "").strip().upper()
                options = pair.get("options", {}) or {}
                allow_e = "E" in options
                is_mcq = allow_e

                question_text = build_question_text(pair)
                inputs, input_len, mode = prepare_inputs(processor, model, image, question_text, dtype)

                gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample)
                if pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = pad_token_id

                with torch.inference_mode():
                    generation = model.generate(**inputs, **gen_kwargs)

                if input_len is not None and generation.shape[-1] > input_len:
                    gen = generation[0][input_len:]
                    decoded = processor.decode(gen, skip_special_tokens=True).strip()
                else:
                    decoded_full = processor.decode(generation[0], skip_special_tokens=True).strip()
                    decoded = decoded_full.splitlines()[-1].strip() if decoded_full else ""

                pred = extract_letter(decoded, allow_e=allow_e)
                is_correct = pred == gt_ans

                # metrics
                metrics["total"] += 1
                metrics["correct"] += int(is_correct)
                metrics["by_type"][q_type]["total"] += 1
                metrics["by_type"][q_type]["correct"] += int(is_correct)

                group = "mcq" if is_mcq else "binary"
                metrics[group]["total"] += 1
                metrics[group]["correct"] += int(is_correct)

                rec = {
                    "id": item.get("id", ""),
                    "q_index": qi,
                    "type": q_type,
                    "is_mcq": is_mcq,
                    "is_trap": bool(pair.get("meta_is_unanswerable", False)),
                    "gt": gt_ans,
                    "pred": pred,
                    "correct": is_correct,
                    "mode": mode,
                    "image_path": ip,
                }
                if args.save_raw:
                    rec["raw"] = decoded
                if args.save_image_abs:
                    rec["image_abs"] = str(img_abs)

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                qa_count += 1
                if args.max_qa is not None and qa_count >= args.max_qa:
                    break

            if args.max_qa is not None and qa_count >= args.max_qa:
                break

    overall = safe_div(metrics["correct"], metrics["total"])
    acc_bin = safe_div(metrics["binary"]["correct"], metrics["binary"]["total"])
    acc_mcq = safe_div(metrics["mcq"]["correct"], metrics["mcq"]["total"])

    sorted_types = sorted(
        metrics["by_type"].items(),
        key=lambda x: safe_div(x[1]["correct"], x[1]["total"]),
        reverse=True,
    )
    type_rows = [
        {"type": qt, "acc": safe_div(s["correct"], s["total"]), "correct": s["correct"], "total": s["total"]}
        for qt, s in sorted_types
    ]

    summary = {
        "model_dir": args.model_dir,
        "model_class": model_class,
        "dataset": str(json_path),
        "run_tag": run_tag,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "stats": {
            "total": metrics["total"],
            "correct": metrics["correct"],
            "overall_accuracy": overall,
            "binary_accuracy": acc_bin,
            "mcq_accuracy": acc_mcq,
            "img_ok": metrics["img_ok"],
            "img_missing": metrics["img_missing"],
            "img_open_failed": metrics["img_open_failed"],
        },
        "details_by_type": type_rows,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if missing_images:
        with open(out_missing, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_images))

    print("\n" + "═" * 60)
    print("📊 Direct Local VLM Evaluation Result")
    print(f"🏆 OVERALL ACCURACY: {overall:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"🔹 Binary (A-D): {acc_bin:.2%}")
    print(f"🔹 MCQ (A-E)   : {acc_mcq:.2%}")
    print(f"🖼️ Image OK: {metrics['img_ok']} | Missing: {metrics['img_missing']} | Open failed: {metrics['img_open_failed']}")
    print("═" * 60)
    print(f"💾 Summary saved to: {out_summary}")
    print(f"💾 Predictions saved to: {out_jsonl}")
    if missing_images:
        print(f"⚠️ Missing image list saved to: {out_missing} (count={len(missing_images)})")


if __name__ == "__main__":
    main()
