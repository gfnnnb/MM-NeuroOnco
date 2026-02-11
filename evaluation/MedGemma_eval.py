#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MedGemma official multimodal flow evaluation (apply_chat_template tokenized).

Open-source friendly:
- CLI configurable model/json/data/out/device/dtype
- Option-E support (A-D / A-E auto by options)
- Optional CUDA_VISIBLE_DEVICES via CLI
- Optional resume by image_path (skip already processed images)
- Outputs JSONL predictions + summary JSON + config JSON
- By default does NOT save raw decoded text or absolute paths (enable flags)
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
from transformers import AutoProcessor, AutoModelForImageTextToText


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a, b):
    return a / b if b else 0.0


def extract_letter(decoded: str, allow_e: bool) -> str:
    """Extract answer letter. allow_e=True => A-E else A-D."""
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


def build_question_text(pair: dict) -> str:
    """Build prompt text; auto A-D vs A-E."""
    options = pair.get("options", {}) or {}
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    allow_e = "E" in options
    valid = "A, B, C, D, or E" if allow_e else "A, B, C, or D"
    return (
        "Answer the multiple-choice question based on the MRI image provided.\n"
        f"Output ONLY one letter: {valid}.\n"
        "Do not output any other text.\n\n"
        f"Question: {pair['question']}\n"
        f"Options:\n{opt_text}\n"
        "Answer:"
    )


def parse_args():
    p = argparse.ArgumentParser(description="MedGemma official apply_chat_template evaluation (Option-E supported).")

    p.add_argument("--model_dir", type=str, required=True, help="Local model dir or HF repo id.")
    p.add_argument("--json_path", type=str, required=True, help="Benchmark JSON path.")
    p.add_argument("--data_root", type=str, required=True, help="Root dir prefix for image_path.")

    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--cuda_visible_devices", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    p.add_argument("--max_new_tokens", type=int, default=10)
    p.add_argument("--do_sample", action="store_true")

    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--max_qa", type=int, default=None)

    p.add_argument("--save_raw", action="store_true", help="Save decoded raw output to JSONL.")
    p.add_argument("--save_image_abs", action="store_true", help="Save absolute image path to JSONL (not recommended).")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"✅ Set CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    model_name_for_tag = Path(args.model_dir.rstrip("/")).name.replace("/", "_")
    run_tag = args.run_tag or f"{model_name_for_tag}_medgemma_official"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / f"preds_{run_tag}.jsonl"
    out_summary = out_dir / f"summary_{run_tag}.json"
    out_config = out_dir / f"config_{run_tag}.json"

    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    json_path = Path(args.json_path)
    data_root = Path(args.data_root)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_images is not None:
        data = data[: args.max_images]

    print(f"📦 Loaded {len(data)} samples from {json_path}")
    print(f"🖼️  data_root={data_root}")
    print(f"🤖 model_dir={args.model_dir}")
    print(f"💾 out_jsonl={out_jsonl}")

    # Load model+processor (official class)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    except Exception as e:
        raise SystemExit(f"❌ Failed to load MedGemma model/processor: {e}")

    done_keys = load_done_keys(out_jsonl) if args.resume else set()
    if args.resume:
        print(f"✅ Resume enabled. Done samples: {len(done_keys)} (by image_path)")

    metrics = {
        "total": 0,
        "correct": 0,
        "binary": {"total": 0, "correct": 0},  # A-D
        "mcq": {"total": 0, "correct": 0},     # with E
        "by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
        "img_missing": 0,
        "img_open_failed": 0,
    }

    qa_count = 0
    write_mode = "a" if args.resume else "w"

    with open(out_jsonl, write_mode, encoding="utf-8") as w:
        for item in tqdm(data, desc="Processing images"):
            ip = item.get("image_path", "")
            if args.resume and ip and ip in done_keys:
                continue

            img_abs = data_root / ip
            if not img_abs.exists():
                metrics["img_missing"] += 1
                continue

            try:
                image = Image.open(img_abs).convert("RGB")
            except Exception:
                metrics["img_open_failed"] += 1
                continue

            for qi, pair in enumerate(item.get("vqa_pairs", []), start=1):
                q_type = pair.get("question_type", "unknown")
                gt = (pair.get("answer") or "").strip().upper()
                options = pair.get("options", {}) or {}
                allow_e = "E" in options
                is_mcq = allow_e

                question_text = build_question_text(pair)

                # Official chat template flow
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question_text},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device, dtype=dtype)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                    )

                # decode only generated part
                gen_ids = generation[0][input_len:]
                decoded = processor.decode(gen_ids, skip_special_tokens=True).strip()

                pred = extract_letter(decoded, allow_e=allow_e)
                ok = pred == gt

                metrics["total"] += 1
                metrics["correct"] += int(ok)
                metrics["by_type"][q_type]["total"] += 1
                metrics["by_type"][q_type]["correct"] += int(ok)

                group = "mcq" if is_mcq else "binary"
                metrics[group]["total"] += 1
                metrics[group]["correct"] += int(ok)

                rec = {
                    "id": item.get("id"),
                    "q_index": qi,
                    "type": q_type,
                    "is_mcq": is_mcq,
                    "is_trap": bool(pair.get("meta_is_unanswerable", False)),
                    "gt": gt,
                    "pred": pred,
                    "correct": ok,
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
        "dataset": str(json_path),
        "run_tag": run_tag,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "stats": {
            "total": metrics["total"],
            "correct": metrics["correct"],
            "overall_accuracy": overall,
            "binary_accuracy": acc_bin,
            "mcq_accuracy": acc_mcq,
            "img_missing": metrics["img_missing"],
            "img_open_failed": metrics["img_open_failed"],
        },
        "details_by_type": type_rows,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 60)
    print("📊 MedGemma (Official apply_chat_template) Evaluation Result")
    print(f"🏆 OVERALL ACCURACY: {overall:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"🔹 Binary (A-D): {acc_bin:.2%}")
    print(f"🔹 MCQ (A-E)   : {acc_mcq:.2%}")
    print(f"🖼️ Missing images: {metrics['img_missing']} | Open failed: {metrics['img_open_failed']}")
    print("═" * 60)
    print(f"💾 Summary saved to: {out_summary}")
    print(f"💾 Predictions saved to: {out_jsonl}")


if __name__ == "__main__":
    main()
