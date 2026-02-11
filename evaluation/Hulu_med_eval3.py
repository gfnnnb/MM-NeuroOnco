#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a local HuggingFace VLM on MM-NeuroOnco closed-ended benchmark.

Open-source friendly:
- No hard-coded private paths
- CLI arguments for model/data/output
- Optional resume from JSONL
- Optional save_raw to store decoded outputs (can be large)
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
from transformers import AutoProcessor, AutoModelForCausalLM


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a, b):
    return a / b if b else 0.0


def extract_letter(decoded: str) -> str:
    """Extract A-E from model decoded text."""
    t = (decoded or "").strip().upper()
    if not t:
        return ""
    m = re.fullmatch(r"([ABCDE])", t)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[\s\(\[\:])([ABCDE])(?:[\.\)\s\:]|$)", t)
    if m:
        return m.group(1)
    m = re.findall(r"[ABCDE]", t)
    return m[-1] if m else ""


def move_to_device_and_fix_dtype(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict:
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            v = v.to(device)
            if k in ("pixel_values", "image", "images") or "pixel_values" in k:
                v = v.to(dtype)
        moved[k] = v
    return moved


def build_prompt_with_chat_template(processor, question: str, options: dict) -> str:
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    valid_choices = "A, B, C, D, or E" if "E" in options else "A, B, C, or D"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Answer the multiple-choice question by outputting ONLY one letter: {valid_choices}.\n"
                        "Do not output any other text.\n\n"
                        f"Question: {question}\n"
                        f"Options:\n{opt_text}\n"
                        "Answer:"
                    ),
                },
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    p = argparse.ArgumentParser(description="Evaluate a local HuggingFace VLM on MM-NeuroOnco (closed-ended).")

    p.add_argument("--model_dir", type=str, required=True, help="Local model directory or HF repo id.")
    p.add_argument("--json_path", type=str, required=True, help="Closed-ended benchmark JSON path.")
    p.add_argument("--data_root", type=str, required=True, help="Root directory of images (prefix for image_path).")

    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--run_tag", type=str, default=None, help="Tag for output filenames.")
    p.add_argument("--resume", action="store_true", help="Resume by skipping samples whose image_path already in JSONL.")

    p.add_argument("--max_qa", type=int, default=None, help="Limit total QA count (for quick tests).")
    p.add_argument("--max_images", type=int, default=None, help="Limit number of images (samples).")

    p.add_argument("--device", type=str, default="cuda", help="Device: cuda / cuda:0 / cpu.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa", help="Attention implementation, e.g. sdpa (if supported).")

    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--save_raw", action="store_true", help="Save decoded raw text into JSONL (can be large).")

    # Optional: enforce CUDA_VISIBLE_DEVICES for your cluster habit
    p.add_argument(
        "--require_cuda_visible_devices",
        type=str,
        default=None,
        help="If set, assert CUDA_VISIBLE_DEVICES equals this string (e.g., '9').",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.require_cuda_visible_devices is not None:
        cur = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert cur == args.require_cuda_visible_devices, (
            f"Please set CUDA_VISIBLE_DEVICES={args.require_cuda_visible_devices} (current={cur})"
        )

    json_path = Path(args.json_path)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name_for_tag = Path(args.model_dir).name.replace("/", "_")
    run_tag = args.run_tag or f"{model_name_for_tag}_closed_eval"

    out_jsonl = out_dir / f"preds_{run_tag}.jsonl"
    out_summary = out_dir / f"summary_{run_tag}.json"
    out_config = out_dir / f"config_{run_tag}.json"

    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")

    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_images is not None:
        data = data[: args.max_images]

    print(f"📦 Loaded {len(data)} samples from {json_path}")
    print(f"🖼️  Image root: {data_root}")
    print(f"🤖 Model: {args.model_dir}")
    print(f"🧠 Device: {device} | dtype={args.dtype} | attn_impl={args.attn_impl}")
    print(f"💾 Outputs: {out_dir}")

    # Load model
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    # Only set _attn_implementation if user provided non-empty
    if args.attn_impl:
        model_kwargs["_attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs).to(device)
    model.eval()

    done_keys = load_done_keys(out_jsonl) if args.resume else set()
    if args.resume:
        print(f"✅ Resume enabled. Done samples: {len(done_keys)} (by image_path)")

    stats = {
        "total": 0,
        "correct": 0,
        "binary": {"total": 0, "correct": 0},
        "mcq": {"total": 0, "correct": 0},
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "img_missing": 0,
        "img_open_failed": 0,
    }

    qa_count = 0
    with open(out_jsonl, "a" if args.resume else "w", encoding="utf-8") as w:
        pbar = tqdm(total=args.max_qa if args.max_qa is not None else None, desc="Inferencing")

        for si, sample in enumerate(data, start=1):
            ip = sample.get("image_path", "")
            if args.resume and ip and ip in done_keys:
                continue

            img_abs = data_root / ip
            if not img_abs.exists():
                stats["img_missing"] += 1
                continue

            try:
                img = Image.open(img_abs).convert("RGB")
            except Exception:
                stats["img_open_failed"] += 1
                continue

            for qi, pair in enumerate(sample.get("vqa_pairs", []), start=1):
                qtype = pair.get("question_type", "unknown")
                gt = (pair.get("answer") or "").strip().upper()
                options = pair.get("options", {}) or {}
                is_mcq = "E" in options

                prompt = build_prompt_with_chat_template(processor, pair["question"], options)
                inputs = processor(images=img, text=prompt, return_tensors="pt")
                inputs = move_to_device_and_fix_dtype(inputs, device=device, dtype=dtype)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )

                decoded = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                pred = extract_letter(decoded)
                ok = pred == gt

                stats["total"] += 1
                stats["correct"] += int(ok)

                stats["by_type"][qtype]["total"] += 1
                stats["by_type"][qtype]["correct"] += int(ok)

                group = "mcq" if is_mcq else "binary"
                stats[group]["total"] += 1
                stats[group]["correct"] += int(ok)

                rec = {
                    "id": sample.get("id", si),
                    "q_index": qi,
                    "type": qtype,
                    "is_mcq": is_mcq,
                    "is_trap": bool(pair.get("meta_is_unanswerable", False)),
                    "gt": gt,
                    "pred": pred,
                    "correct": ok,
                    "image_path": ip,
                }
                if args.save_raw:
                    rec["raw"] = decoded

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                qa_count += 1
                if args.max_qa is None:
                    pbar.update(1)
                else:
                    pbar.update(1)
                    if qa_count >= args.max_qa:
                        pbar.close()
                        break

            if args.max_qa is not None and qa_count >= args.max_qa:
                break

        pbar.close()

    # Summary
    overall_acc = safe_div(stats["correct"], stats["total"])
    acc_bin = safe_div(stats["binary"]["correct"], stats["binary"]["total"])
    acc_mcq = safe_div(stats["mcq"]["correct"], stats["mcq"]["total"])

    sorted_types = sorted(
        stats["by_type"].items(),
        key=lambda x: safe_div(x[1]["correct"], x[1]["total"]),
        reverse=True,
    )
    type_rows = []
    for t, s in sorted_types:
        type_rows.append(
            {
                "type": t,
                "acc": safe_div(s["correct"], s["total"]),
                "correct": s["correct"],
                "total": s["total"],
            }
        )

    summary = {
        "model": args.model_dir,
        "dataset": str(json_path),
        "run_tag": run_tag,
        "device": str(device),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "stats": {
            "total": stats["total"],
            "correct": stats["correct"],
            "overall_accuracy": overall_acc,
            "binary_accuracy": acc_bin,
            "mcq_accuracy": acc_mcq,
            "img_missing": stats["img_missing"],
            "img_open_failed": stats["img_open_failed"],
        },
        "details_by_type": type_rows,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 60)
    print("📊 Local HF VLM Evaluation Result")
    print(f"🏆 OVERALL ACCURACY: {overall_acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"🔹 Binary Tasks: {acc_bin:.2%}")
    print(f"🔹 MCQ Tasks   : {acc_mcq:.2%}")
    print(f"🖼️ Missing images: {stats['img_missing']} | Open failed: {stats['img_open_failed']}")
    print("═" * 60)
    print(f"💾 Summary saved to: {out_summary}")
    print(f"💾 Predictions saved to: {out_jsonl}")


if __name__ == "__main__":
    main()
