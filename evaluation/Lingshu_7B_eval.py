#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified local VLM evaluator for MM-NeuroOnco closed-ended benchmark.

Key features:
- CLI configurable (model/json/data/output/device/dtype)
- Optional resume by image_path
- Robust input preparation (3-stage fallback):
  1) tokenized chat template
  2) prompt string from chat template
  3) plain (text, images) processor call
- Output:
  - JSONL predictions (appendable)
  - Summary JSON
  - Missing image log
- Open-source friendly: no private hard-coded paths, no forced CUDA_VISIBLE_DEVICES
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


def extract_letter(decoded: str) -> str:
    """Extract A-E."""
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


def resolve_image_path(data_root: Path, image_path_str: str, extra_roots=None):
    """
    Resolve image_path to an existing file:
    1) data_root / image_path
    2) strip possible duplicated root directory segment
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
    Generic VLM loader: try common multimodal model classes in order.
    """
    print(f"🔧 Loading processor: {model_dir}")
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
            # Some models accept _attn_implementation; safe to try only if provided
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


def build_prompt_text(question: str, options: dict) -> str:
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    valid_choices = "A, B, C, D, or E" if "E" in options else "A, B, C, or D"
    return (
        f"Answer the multiple-choice question by outputting ONLY one letter: {valid_choices}.\n"
        "Do not output any other text.\n\n"
        f"Question: {question}\n"
        f"Options:\n{opt_text}\n"
        "Answer:"
    )


def move_pixel_values_to_dtype(inputs: dict, dtype: torch.dtype):
    for k in list(inputs.keys()):
        v = inputs[k]
        if hasattr(v, "to") and ("pixel_values" in k):
            inputs[k] = v.to(dtype)
    return inputs


def prepare_inputs(processor, model, image: Image.Image, question: str, options: dict, dtype: torch.dtype):
    """
    3-stage fallback:
    1) tokenized chat template (best)
    2) prompt string + processor(images,text)
    3) processor(text,images)
    Returns: inputs, input_len, mode
    """
    prompt_text = build_prompt_text(question, options)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 1) tokenized template
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

    # 2) prompt string template
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        inputs = move_pixel_values_to_dtype(inputs, dtype)
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else None
        return inputs, input_len, "chat_template_prompt"
    except Exception:
        pass

    # 3) fallback
    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs = move_pixel_values_to_dtype(inputs, dtype)
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
    p = argparse.ArgumentParser(description="Unified local VLM evaluator for MM-NeuroOnco closed-ended benchmark.")

    p.add_argument("--model_dir", type=str, required=True, help="Local model directory or HF repo id.")
    p.add_argument("--json_path", type=str, required=True, help="Benchmark JSON path.")
    p.add_argument("--data_root", type=str, required=True, help="Root dir that prefixes image_path.")

    p.add_argument("--extra_roots", type=str, nargs="*", default=[], help="Optional extra roots for resolving images.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--run_tag", type=str, default=None, help="Tag used in output filenames.")

    p.add_argument("--resume", action="store_true", help="Resume by skipping image_path already in JSONL.")
    p.add_argument("--max_images", type=int, default=None, help="Limit number of samples/images.")
    p.add_argument("--max_qa", type=int, default=None, help="Limit total QA count across all samples.")

    p.add_argument("--cuda_visible_devices", type=str, default=None, help="If set, export CUDA_VISIBLE_DEVICES.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default=None, help="Attention impl (e.g., sdpa) if supported.")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--do_sample", action="store_true", help="Use sampling decoding (default: greedy).")

    p.add_argument("--save_raw", action="store_true", help="Save decoded text in JSONL (can be large).")
    p.add_argument("--save_image_abs", action="store_true", help="Save absolute resolved image path (may leak).")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"✅ Set CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    json_path = Path(args.json_path)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name_for_tag = Path(args.model_dir.rstrip("/")).name.replace("/", "_")
    run_tag = args.run_tag or f"{model_name_for_tag}_local_vlm"

    out_jsonl = out_dir / f"preds_{run_tag}.jsonl"
    out_summary = out_dir / f"summary_{run_tag}.json"
    out_missing = out_dir / f"missing_{run_tag}.txt"
    out_config = out_dir / f"config_{run_tag}.json"

    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_images is not None:
        data = data[: args.max_images]

    print(f"📦 Samples: {len(data)}")
    print(f"🖼️  data_root: {data_root}")
    print(f"🤖 model_dir: {args.model_dir}")
    print(f"💾 out_dir: {out_dir}")
    print(f"🧾 out_jsonl: {out_jsonl}")

    model, processor, loaded_cls = load_vlm(args.model_dir, dtype, args.attn_impl)
    pad_token_id = get_pad_token_id(processor)
    if pad_token_id is not None:
        print(f"✅ pad_token_id={pad_token_id}")
    print(f"✅ model class: {loaded_cls}")

    done_keys = load_done_keys(out_jsonl) if args.resume else set()
    if args.resume:
        print(f"✅ Resume enabled. Done samples: {len(done_keys)} (by image_path)")

    stats = {
        "total": 0,
        "correct": 0,
        "binary": {"total": 0, "correct": 0},
        "mcq": {"total": 0, "correct": 0},
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "img_ok": 0,
        "img_missing": 0,
        "img_open_failed": 0,
    }

    missing_images = []
    qa_count = 0

    extra_roots = [Path(x) for x in args.extra_roots]

    write_mode = "a" if args.resume else "w"
    with open(out_jsonl, write_mode, encoding="utf-8") as w:
        pbar = tqdm(total=len(data), desc="Processing samples")

        for sample in data:
            ip = sample.get("image_path", "")
            if args.resume and ip and ip in done_keys:
                pbar.update(1)
                continue

            img_abs = resolve_image_path(data_root, ip, extra_roots=extra_roots)
            if img_abs is None:
                stats["img_missing"] += 1
                missing_images.append(ip)
                pbar.update(1)
                continue

            try:
                img = Image.open(img_abs).convert("RGB")
                stats["img_ok"] += 1
            except Exception as e:
                stats["img_open_failed"] += 1
                missing_images.append(f"{ip} | open_failed: {e}")
                pbar.update(1)
                continue

            for qi, pair in enumerate(sample.get("vqa_pairs", []), start=1):
                qtype = pair.get("question_type", "unknown")
                gt = (pair.get("answer") or "").strip().upper()
                options = pair.get("options", {}) or {}
                is_mcq = "E" in options

                inputs, input_len, mode = prepare_inputs(processor, model, img, pair["question"], options, dtype)

                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                }
                if pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = pad_token_id

                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)

                # Try to slice off prompt
                if input_len is not None and out.shape[-1] > input_len:
                    decoded = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
                else:
                    decoded_full = processor.decode(out[0], skip_special_tokens=True).strip()
                    decoded = decoded_full.splitlines()[-1].strip() if decoded_full else ""

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
                    "id": sample.get("id", ""),
                    "q_index": qi,
                    "type": qtype,
                    "is_mcq": is_mcq,
                    "is_trap": bool(pair.get("meta_is_unanswerable", False)),
                    "gt": gt,
                    "pred": pred,
                    "correct": ok,
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

            pbar.update(1)

            if args.max_qa is not None and qa_count >= args.max_qa:
                break

        pbar.close()

    overall_acc = safe_div(stats["correct"], stats["total"])
    acc_bin = safe_div(stats["binary"]["correct"], stats["binary"]["total"])
    acc_mcq = safe_div(stats["mcq"]["correct"], stats["mcq"]["total"])

    sorted_types = sorted(
        stats["by_type"].items(),
        key=lambda x: safe_div(x[1]["correct"], x[1]["total"]),
        reverse=True,
    )
    type_rows = [
        {
            "type": qt,
            "acc": safe_div(s["correct"], s["total"]),
            "correct": s["correct"],
            "total": s["total"],
        }
        for qt, s in sorted_types
    ]

    summary = {
        "model_dir": args.model_dir,
        "model_class": loaded_cls,
        "dataset": str(json_path),
        "run_tag": run_tag,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "stats": {
            "total": stats["total"],
            "correct": stats["correct"],
            "overall_accuracy": overall_acc,
            "binary_accuracy": acc_bin,
            "mcq_accuracy": acc_mcq,
            "img_ok": stats["img_ok"],
            "img_missing": stats["img_missing"],
            "img_open_failed": stats["img_open_failed"],
        },
        "details_by_type": type_rows,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if missing_images:
        with open(out_missing, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_images))

    print("\n" + "═" * 60)
    print("📊 Unified Local VLM Evaluation Result")
    print(f"🏆 OVERALL ACCURACY: {overall_acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"🔹 Binary Tasks: {acc_bin:.2%}")
    print(f"🔹 MCQ Tasks   : {acc_mcq:.2%}")
    print(f"🖼️ Image OK: {stats['img_ok']} | Missing: {stats['img_missing']} | Open failed: {stats['img_open_failed']}")
    print("═" * 60)
    print(f"💾 Summary saved to: {out_summary}")
    print(f"💾 Predictions saved to: {out_jsonl}")
    if missing_images:
        print(f"⚠️ Missing image list saved to: {out_missing} (count={len(missing_images)})")


if __name__ == "__main__":
    main()
