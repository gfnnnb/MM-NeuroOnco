#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot baseline evaluation for Qwen3-VL style VQA benchmark.

Open-source friendly changes:
- Remove private absolute paths
- Use CLI args / env vars
- Do NOT hard-code CUDA_VISIBLE_DEVICES
- Keep optional pixel limits configurable
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq


def extract_letter(decoded: str) -> str:
    """Extract a valid option letter A-E from model output."""
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


def build_prompt(question: str, options: dict) -> str:
    """Build prompt in a consistent evaluation style."""
    keys = [k for k in ["A", "B", "C", "D", "E"] if k in options]
    opt_text = "\n".join([f"{k}. {options[k]}" for k in keys])

    return (
        f"{question}\n\n"
        f"Options:\n{opt_text}\n\n"
        "Answer with the option letter only."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        default=os.environ.get("MMNO_DATA_ROOT", "."),
        help="Root directory that contains images referenced by image_path in JSON.",
    )
    p.add_argument(
        "--json_path",
        default=os.environ.get("MMNO_BENCH_JSON", "Final_Benchmark_VQA_Shuffled_E.json"),
        help="Benchmark JSON path (list of items with image_path and vqa_pairs).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MMNO_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        help="HF model id or local path.",
    )

    p.add_argument("--output_stats", default="Baseline_Qwen3VL_ZeroShot_Stats.json")
    p.add_argument("--output_preds", default="Baseline_Qwen3VL_ZeroShot_Preds.jsonl")

    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (default off).")

    p.add_argument("--use_pixel_limits", action="store_true", help="Enable pixel limits.")
    p.add_argument("--min_pixels", type=int, default=256 * 256)
    p.add_argument("--max_pixels", type=int, default=512 * 512)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    dtype = resolve_dtype(args.dtype)

    print("🌟 Starting Zero-shot Evaluation")
    print(f"📂 data_root : {args.data_root}")
    print(f"🧾 json_path : {args.json_path}")
    print(f"🧠 model     : {args.model}")
    print(f"🖥️ device    : {device} | dtype={args.dtype}")

    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON not found: {args.json_path}")

    # Load model/processor
    if args.use_pixel_limits:
        processor = AutoProcessor.from_pretrained(
            args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            trust_remote_code=True,
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype if device.type == "cuda" else None,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Model loaded.")

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list.")

    stats = {
        "overall": {"correct": 0, "total": 0},
        "4_options": {"correct": 0, "total": 0},
        "5_options": {"correct": 0, "total": 0},
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
    }

    pred_f = open(args.output_preds, "w", encoding="utf-8")

    print(f"📝 Testing {len(data)} images...")

    for item in tqdm(data, desc="Inference"):
        img_rel = item.get("image_path", "")
        img_abs = os.path.join(args.data_root, img_rel)

        if not os.path.exists(img_abs):
            pred_f.write(json.dumps(
                {"image_path": img_rel, "error": f"missing_image: {img_abs}"},
                ensure_ascii=False
            ) + "\n")
            continue

        try:
            image = Image.open(img_abs).convert("RGB")
        except Exception as e:
            pred_f.write(json.dumps(
                {"image_path": img_rel, "error": f"corrupt_image: {str(e)}"},
                ensure_ascii=False
            ) + "\n")
            continue

        for pair in item.get("vqa_pairs", []):
            q_type = pair.get("question_type", "unknown")
            gt_ans = (pair.get("answer", "") or "").strip().upper()
            options = pair.get("options", {}) or {}

            group_key = "5_options" if "E" in options else "4_options"

            prompt_text = build_prompt(pair.get("question", ""), options)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

            if device.type == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            pred = extract_letter(decoded)
            is_correct = (pred == gt_ans)

            # Stats
            stats["overall"]["total"] += 1
            stats[group_key]["total"] += 1
            stats["by_type"][q_type]["total"] += 1
            if is_correct:
                stats["overall"]["correct"] += 1
                stats[group_key]["correct"] += 1
                stats["by_type"][q_type]["correct"] += 1

            pred_f.write(json.dumps({
                "image_path": img_rel,
                "question_type": q_type,
                "question": pair.get("question", ""),
                "options": options,
                "gt": gt_ans,
                "pred": pred,
                "raw": decoded,
                "correct": bool(is_correct),
            }, ensure_ascii=False) + "\n")

    pred_f.close()

    acc = stats["overall"]["correct"] / (stats["overall"]["total"] or 1)
    acc4 = stats["4_options"]["correct"] / (stats["4_options"]["total"] or 1)
    acc5 = stats["5_options"]["correct"] / (stats["5_options"]["total"] or 1)

    print("\n" + "═" * 60)
    print("📊 BASELINE RESULTS")
    print(f"🏆 Overall: {acc:.2%}  ({stats['overall']['correct']}/{stats['overall']['total']})")
    print(f"🔹 4-Opt : {acc4:.2%}  ({stats['4_options']['correct']}/{stats['4_options']['total']})")
    print(f"🔹 5-Opt : {acc5:.2%}  ({stats['5_options']['correct']}/{stats['5_options']['total']})")
    print("═" * 60)

    stats_dump = {
        "overall": stats["overall"],
        "4_options": stats["4_options"],
        "5_options": stats["5_options"],
        "by_type": {k: v for k, v in stats["by_type"].items()},
    }

    with open(args.output_stats, "w", encoding="utf-8") as f:
        json.dump({"stats": stats_dump}, f, ensure_ascii=False, indent=2)

    print(f"🧾 Saved stats: {args.output_stats}")
    print(f"🧾 Saved preds: {args.output_preds}")


if __name__ == "__main__":
    main()
