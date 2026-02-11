#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch inference for Open-ended VQA benchmark with Qwen3-VL + LoRA adapter.

Open-source friendly changes:
- No hard-coded CUDA_VISIBLE_DEVICES / no single-GPU assertion
- No private absolute paths
- Use CLI args / env vars for model, adapter, data root, json path
- Robust handling of missing/corrupt images (skip but log)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL Open Benchmark Batch Inference (LoRA)")

    p.add_argument("--model", default=os.environ.get("MMNO_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
                   help="HF model id or local path for base model.")
    p.add_argument("--adapter", default=os.environ.get("MMNO_ADAPTER", ""),
                   help="Path to LoRA adapter directory (PEFT). Required.")
    p.add_argument("--json_path", default=os.environ.get("MMNO_OPEN_BENCH_JSON", "Final_Benchmark_Open_Filtered.json"),
                   help="Open benchmark JSON path (items with image_path and qa_pairs).")
    p.add_argument("--data_root", default=os.environ.get("MMNO_DATA_ROOT", "."),
                   help="Root directory for image_path resolution.")

    p.add_argument("--output", default="preds_open_lora.jsonl", help="Output JSONL file.")
    p.add_argument("--test", action="store_true", help="Test mode: run first 2 batches only.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=8, help="ThreadPool workers for image I/O.")
    p.add_argument("--merge_lora", action="store_true",
                   help="Try merge_and_unload() for faster inference (optional).")

    p.add_argument("--skip_missing_images", action="store_true",
                   help="Skip tasks with missing/corrupt images (default: True behavior).")

    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def resolve_image_path(data_root: str, rel_path: str) -> Optional[str]:
    if not rel_path:
        return None
    p = os.path.join(data_root, rel_path)
    return p if os.path.exists(p) else None


def prepare_single_sample(
    task: Tuple[Dict[str, Any], Dict[str, Any]],
    processor: Any,
    data_root: str
) -> Dict[str, Any]:
    """Load image + build prompt (question only) in a thread."""
    item, qa = task

    img_abs = resolve_image_path(data_root, item.get("image_path", ""))
    image = None
    if img_abs:
        try:
            image = Image.open(img_abs).convert("RGB")
        except Exception:
            image = None

    question = qa.get("question", "")

    if image is not None:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
    else:
        messages = [{"role": "user", "content": question}]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return {
        "text_prompt": text_prompt,
        "image": image,
        "meta": {
            "id": item.get("id", None),
            "question_id": qa.get("question_id", None),
            "question_type": qa.get("question_type", None),
            "question": question,
            "reference_answer": qa.get("reference_answer", ""),
            "eval_keywords": qa.get("eval_keywords", []),
            "image_path": item.get("image_path", ""),
        }
    }


def main():
    args = parse_args()

    if not args.adapter:
        raise ValueError("--adapter is required (path to LoRA adapter directory).")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    dtype = resolve_dtype(args.dtype)

    print("\n🚀 Qwen3-VL Open Benchmark (LoRA)")
    print(f"🧠 Base model : {args.model}")
    print(f"🧩 Adapter    : {args.adapter}")
    print(f"📂 Data root  : {args.data_root}")
    print(f"🧾 JSON path  : {args.json_path}")
    print(f"🖥️ Device     : {device} | dtype={args.dtype}")
    print(f"📦 Batch      : {args.batch_size} | max_new_tokens={args.max_new_tokens}")

    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"Benchmark JSON not found: {args.json_path}")

    # 1) processor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2) base model
    print("\n🌟 Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype if device.type == "cuda" else None,
        device_map={"": "cuda:0"} if device.type == "cuda" else None,
    )

    # 3) LoRA
    print("🧩 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    if args.merge_lora:
        try:
            model = model.merge_and_unload()
            print("✅ LoRA merged into base model.")
        except Exception as e:
            print(f"⚠️ merge_and_unload failed (running unmerged): {e}")

    model.eval()
    print("✅ Model ready.")

    # 4) load benchmark and flatten tasks
    with open(args.json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for item in raw_data:
        for qa in item.get("qa_pairs", []):
            tasks.append((item, qa))

    print(f"📦 Total Open Questions: {len(tasks)}")

    # 5) output
    if os.path.exists(args.output):
        print(f"⚠️ Output exists and will be overwritten: {args.output}")

    out_f = open(args.output, "w", encoding="utf-8")

    # 6) batch inference
    io_pool = ThreadPoolExecutor(max_workers=args.num_workers)
    batch_size = args.batch_size
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    if args.test:
        total_batches = min(2, total_batches)

    pbar = tqdm(total=total_batches, desc="Batch Inference (LoRA)")

    for bi, start in enumerate(range(0, len(tasks), batch_size)):
        if args.test and bi >= 2:
            break

        batch_tasks = tasks[start: start + batch_size]
        batch_results = list(io_pool.map(
            lambda t: prepare_single_sample(t, processor, args.data_root),
            batch_tasks
        ))

        prompts = [r["text_prompt"] for r in batch_results]
        images = [r["image"] for r in batch_results]
        metas = [r["meta"] for r in batch_results]

        valid_idx = [i for i, img in enumerate(images) if img is not None]
        if not valid_idx:
            # optionally write errors
            if not args.skip_missing_images:
                for m in metas:
                    m["prediction"] = ""
                    m["error"] = "missing_or_corrupt_image"
                    out_f.write(json.dumps(m, ensure_ascii=False) + "\n")
                out_f.flush()
            pbar.update(1)
            continue

        final_prompts = [prompts[i] for i in valid_idx]
        final_images = [images[i] for i in valid_idx]
        final_metas = [metas[i] for i in valid_idx]

        inputs = processor(
            text=final_prompts,
            images=final_images,
            padding=True,
            return_tensors="pt"
        )

        if device.type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for i, text in enumerate(decoded):
                record = final_metas[i]
                record["prediction"] = (text or "").strip()

                if args.test and i == 0:
                    print("-" * 30)
                    print(f"QID:  {record.get('question_id')}")
                    print(f"Type: {record.get('question_type')}")
                    print(f"Q:    {record.get('question')}")
                    ref = (record.get("reference_answer") or "")
                    print(f"REF:  {ref[:200]}{'...' if len(ref)>200 else ''}")
                    print(f"PRED: {record['prediction'][:200]}{'...' if len(record['prediction'])>200 else ''}")
                    print(f"KW:   {record.get('eval_keywords')}")

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            out_f.flush()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ OOM: batch_size={batch_size} too large. Try --batch_size smaller.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"\n❌ Runtime error: {e}")
        except Exception as e:
            print(f"\n❌ Unknown error: {e}")

        pbar.update(1)

    io_pool.shutdown()
    out_f.close()
    pbar.close()

    print(f"\n✅ Done! Saved to: {args.output}")


if __name__ == "__main__":
    main()
