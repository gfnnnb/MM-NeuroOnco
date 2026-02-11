#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MM-NeuroOnco closed-ended benchmark evaluator (open-source friendly).

Key features:
- No hard-coded API keys / private base URLs
- CLI arguments for paths and run settings
- Resume from existing JSONL
- Concurrent evaluation with retry/backoff
- Periodic summary saving
"""

import os
import json
import re
import base64
import time
import random
import threading
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from PIL import Image

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit(
        "Missing dependency: openai. Install with: pip install openai"
    ) from e


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a, b):
    return a / b if b else 0.0


def resolve_image_path(data_root: Path, image_path_str: str, extra_roots=None):
    """
    Robust image path resolution:
    1) data_root / image_path
    2) if image_path contains data_root.name, strip prefix and retry
    3) try each extra_root with same logic
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


def to_data_url(img_path: Path) -> str:
    suffix = img_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_multi_question_prompt(vqa_pairs):
    blocks = []
    for i, pair in enumerate(vqa_pairs, start=1):
        options = pair.get("options", {}) or {}
        opt_text = "\n".join([f"({k}) {v}" for k, v in options.items()])
        valid = "A, B, C, D, or E" if "E" in options else "A, B, C, or D"
        blocks.append(
            f"Question {i} (Select {valid}):\n{pair['question']}\nOptions:\n{opt_text}\n"
        )

    return (
        "You are an expert radiologist analyzing an MRI image.\n"
        "You will be presented with multiple multiple-choice questions.\n\n"
        "RULES:\n"
        "1. Analyze the image carefully.\n"
        "2. Answer ALL questions in order.\n"
        "3. Output strictly in the following format:\n"
        "Q1: <LETTER>\n"
        "Q2: <LETTER>\n"
        "...\n"
        "4. Do NOT output explanations, reasoning, or any other text.\n\n"
        "Questions:\n"
        + "\n".join(blocks)
        + "\nYour Answers:\n"
    )


def parse_batch_answers(text: str, n: int):
    """
    Robust parser:
    1) line-based "Q1: A"
    2) global regex search (handles one-line output)
    3) fallback to letter extraction if count matches
    """
    ans = [""] * n
    text = text.replace("```json", "").replace("```", "").strip()

    # Strategy 1: line matches
    for line in text.splitlines():
        m = re.match(
            r"^\s*(?:Q|Question\s)?(\d+)\s*[:：\.]\s*([ABCDE])\b",
            line.strip(),
            flags=re.I,
        )
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                ans[idx] = m.group(2).upper()

    # Strategy 2: global regex
    if any(a == "" for a in ans):
        matches = re.findall(
            r"(?:Q|Question\s)?(\d+)\s*[:：\.]\s*([ABCDE])\b", text, flags=re.I
        )
        for qnum, letter in matches:
            idx = int(qnum) - 1
            if 0 <= idx < n:
                ans[idx] = letter.upper()

    # Strategy 3: fallback letters (careful!)
    if any(a == "" for a in ans):
        letters = re.findall(r"\b([ABCDE])\b", text.upper())
        if len(letters) == n:
            ans = letters
        elif len(letters) > n:
            ans = letters[:n]

    return ans


def call_api_with_retry(
    client: OpenAI,
    model: str,
    messages,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    base_backoff: float,
):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip(), None
        except Exception as e:
            last_err = e
            sleep_s = base_backoff * (2**attempt) + random.uniform(0.5, 1.5)
            if attempt > 1:
                print(f"⚠️ Retry {attempt + 1}/{max_retries} due to: {e}")
            time.sleep(sleep_s)
    return "", last_err


def load_done_keys(jsonl_path: Path):
    """
    Resume key uses `image_path` by default (same as original script).
    """
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


def merge_partial(stats, partial):
    stats["total"] += partial["total"]
    stats["correct"] += partial["correct"]
    stats["binary"]["total"] += partial["binary_total"]
    stats["binary"]["correct"] += partial["binary_correct"]
    stats["mcq"]["total"] += partial["mcq_total"]
    stats["mcq"]["correct"] += partial["mcq_correct"]
    stats["trap"]["total"] += partial["trap_total"]
    stats["trap"]["correct"] += partial["trap_correct"]
    stats["non_trap"]["total"] += partial["non_trap_total"]
    stats["non_trap"]["correct"] += partial["non_trap_correct"]
    stats["img_ok"] += partial["img_ok"]
    stats["img_missing"] += partial["img_missing"]
    stats["img_open_failed"] += partial["img_open_failed"]
    stats["api_failed"] += partial["api_failed"]
    stats["api_calls"] += partial["api_calls"]
    for t, s in partial["by_type"].items():
        stats["by_type"][t]["total"] += s["total"]
        stats["by_type"][t]["correct"] += s["correct"]


def build_summary(stats, total_samples, processed_samples, model, run_tag):
    overall_acc = safe_div(stats["correct"], stats["total"])
    bin_acc = safe_div(stats["binary"]["correct"], stats["binary"]["total"])
    mcq_acc = safe_div(stats["mcq"]["correct"], stats["mcq"]["total"])
    trap_acc = safe_div(stats["trap"]["correct"], stats["trap"]["total"])
    non_trap_acc = safe_div(stats["non_trap"]["correct"], stats["non_trap"]["total"])

    sorted_types = sorted(
        stats["by_type"].items(),
        key=lambda x: safe_div(x[1]["correct"], x[1]["total"]),
        reverse=True,
    )
    type_rows = [
        {
            "type": t,
            "acc": safe_div(s["correct"], s["total"]),
            "correct": s["correct"],
            "total": s["total"],
        }
        for t, s in sorted_types
    ]

    return {
        "model": model,
        "run_tag": run_tag,
        "total_samples": total_samples,
        "processed": processed_samples,
        "stats": stats,
        "overall_accuracy": overall_acc,
        "binary_accuracy": bin_acc,
        "mcq_accuracy": mcq_acc,
        "trap_accuracy": trap_acc,
        "non_trap_accuracy": non_trap_acc,
        "details_by_type": type_rows,
    }


def process_one_sample(
    sample,
    si,
    data_root: Path,
    extra_roots,
    client: OpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    base_backoff: float,
):
    partial = {
        "total": 0,
        "correct": 0,
        "binary_total": 0,
        "binary_correct": 0,
        "mcq_total": 0,
        "mcq_correct": 0,
        "trap_total": 0,
        "trap_correct": 0,
        "non_trap_total": 0,
        "non_trap_correct": 0,
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "img_ok": 0,
        "img_missing": 0,
        "img_open_failed": 0,
        "api_failed": 0,
        "api_calls": 0,
    }

    img_abs = resolve_image_path(data_root, sample["image_path"], extra_roots=extra_roots)
    if img_abs is None:
        partial["img_missing"] += 1
        return [], partial

    try:
        Image.open(img_abs).convert("RGB")
        partial["img_ok"] += 1
    except Exception:
        partial["img_open_failed"] += 1
        return [], partial

    img_data_url = to_data_url(img_abs)
    vqa_pairs = sample["vqa_pairs"]
    prompt = build_multi_question_prompt(vqa_pairs)

    messages = [
        {"role": "system", "content": "You are an expert radiologist. Answer succinctly."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_data_url}},
            ],
        },
    ]

    partial["api_calls"] += 1
    raw_batch, err = call_api_with_retry(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        base_backoff=base_backoff,
    )

    if err is not None:
        partial["api_failed"] += 1
        raw_batch = f"API_ERROR: {err}"

    preds = parse_batch_answers(raw_batch, n=len(vqa_pairs))

    records = []
    for i, pair in enumerate(vqa_pairs):
        qtype = pair.get("question_type", "unknown")
        gt = (pair.get("answer") or "").strip().upper()
        options = pair.get("options", {}) or {}
        is_mcq = "E" in options
        is_trap = bool(pair.get("meta_is_unanswerable", False))

        pred = preds[i] if i < len(preds) else ""
        ok = pred == gt

        partial["total"] += 1
        if ok:
            partial["correct"] += 1
        partial["by_type"][qtype]["total"] += 1
        if ok:
            partial["by_type"][qtype]["correct"] += 1

        if is_mcq:
            partial["mcq_total"] += 1
            if ok:
                partial["mcq_correct"] += 1
        else:
            partial["binary_total"] += 1
            if ok:
                partial["binary_correct"] += 1

        if is_trap:
            partial["trap_total"] += 1
            if ok:
                partial["trap_correct"] += 1
        else:
            partial["non_trap_total"] += 1
            if ok:
                partial["non_trap_correct"] += 1

        records.append(
            {
                "id": sample.get("id", si),
                "q_index": i + 1,
                "type": qtype,
                "gt": gt,
                "pred": pred,
                "raw_batch": raw_batch,
                "correct": ok,
                "image_path": sample.get("image_path", ""),
            }
        )

    return records, partial


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MM-NeuroOnco closed-ended VQA benchmark (multi-question per image)."
    )
    p.add_argument("--json_path", type=str, required=True, help="Path to closed-ended benchmark JSON.")
    p.add_argument("--data_root", type=str, required=True, help="Root directory containing images.")
    p.add_argument(
        "--extra_roots",
        type=str,
        nargs="*",
        default=[],
        help="Optional extra roots to search for images (space-separated).",
    )

    p.add_argument("--model", type=str, default=os.environ.get("MMNO_MODEL", "gpt-4.1-mini"))
    p.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", help="Env var name for API key.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=512)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=10)
    p.add_argument("--base_backoff", type=float, default=1.0)

    p.add_argument("--run_tag", type=str, default=None, help="Optional run tag (used in output filenames).")
    p.add_argument("--out_dir", type=str, default="outputs", help="Directory to save predictions and summaries.")
    p.add_argument("--save_every", type=int, default=50, help="Save summary every N processed samples.")

    return p.parse_args()


def main():
    args = parse_args()

    # API key via env (no defaults!)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Missing API key: set environment variable {args.api_key_env} before running."
        )

    json_path = Path(args.json_path)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    run_tag = args.run_tag or f"{model_name}_closed_eval"

    out_jsonl = out_dir / f"preds_{run_tag}.jsonl"
    out_summary = out_dir / f"summary_{run_tag}.json"
    out_config = out_dir / f"config_{run_tag}.json"

    # Save config for reproducibility
    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Thread-local OpenAI client
    thread_local = threading.local()

    def get_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = OpenAI(api_key=api_key, base_url=args.base_url)
        return thread_local.client

    data = json.load(open(json_path, "r", encoding="utf-8"))
    total_samples = len(data)

    done_keys = load_done_keys(out_jsonl)
    print(f"✅ Resume enabled. Done: {len(done_keys)}")

    todo = []
    for si, sample in enumerate(data, start=1):
        ip = sample.get("image_path", "")
        if ip and ip in done_keys:
            continue
        todo.append((si, sample))

    print(f"📦 Total samples: {total_samples} | Todo: {len(todo)}")
    print(f"🤖 Model: {model_name} | Workers: {args.workers}")
    print(f"💾 Output: {out_dir}")

    if not todo:
        print("🎉 All done!")
        return

    stats = {
        "total": 0,
        "correct": 0,
        "binary": {"total": 0, "correct": 0},
        "mcq": {"total": 0, "correct": 0},
        "trap": {"total": 0, "correct": 0},
        "non_trap": {"total": 0, "correct": 0},
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "img_ok": 0,
        "img_missing": 0,
        "img_open_failed": 0,
        "api_failed": 0,
        "api_calls": 0,
    }

    processed_samples = 0
    write_lock = threading.Lock()

    # Append mode for resume
    with open(out_jsonl, "a", encoding="utf-8") as w:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    process_one_sample,
                    sample,
                    si,
                    data_root,
                    args.extra_roots,
                    get_client(),
                    model_name,
                    args.max_tokens,
                    args.temperature,
                    args.max_retries,
                    args.base_backoff,
                )
                for si, sample in todo
            ]

            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Eval {model_name}")
            for fut in pbar:
                records, partial = fut.result()

                if records:
                    with write_lock:
                        for r in records:
                            w.write(json.dumps(r, ensure_ascii=False) + "\n")
                        w.flush()

                merge_partial(stats, partial)
                processed_samples += 1

                curr_acc = safe_div(stats["correct"], stats["total"])
                pbar.set_postfix({"Acc": f"{curr_acc:.2%}", "Fail": stats["api_failed"]})

                if processed_samples % args.save_every == 0:
                    summary = build_summary(stats, total_samples, processed_samples, model_name, run_tag)
                    with open(out_summary, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary = build_summary(stats, total_samples, processed_samples, model_name, run_tag)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 60)
    print(f"📊 {model_name} FINISHED")
    print(f"API Failed: {stats['api_failed']}")
    print(f"🏆 OVERALL ACCURACY: {summary['overall_accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print("═" * 60)


if __name__ == "__main__":
    main()
