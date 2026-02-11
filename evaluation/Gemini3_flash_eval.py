#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MM-NeuroOnco closed-ended evaluator (single-question-per-call).

Features:
- Open-source friendly: no hard-coded API keys / private URLs / private paths
- CLI arguments for dataset & output paths
- Resume from JSONL
- Concurrent processing across images; sequential calls per question within an image
- Periodic summary writing + post-hoc JSONL summary for "this run only"
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
    raise SystemExit("Missing dependency: openai. Install with: pip install openai") from e


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a, b):
    return a / b if b else 0.0


def resolve_image_path(data_root: Path, image_path_str: str, extra_roots=None):
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


def build_single_question_prompt(pair, q_index: int):
    options = pair.get("options", {}) or {}
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    valid = "A, B, C, D, or E" if "E" in options else "A, B, C, or D"

    return (
        "You will be given an MRI image and ONE multiple-choice question about it.\n"
        f"Answer with exactly ONE letter: {valid}.\n"
        "Output MUST be exactly:\n"
        "<LETTER>\n"
        "Do NOT output any other text.\n\n"
        f"Q{q_index} ({valid}): {pair['question']}\n"
        f"Options:\n{opt_text}\n\n"
        "Answer:\n"
    )


def parse_single_answer(text: str):
    if not text:
        return ""
    t = text.strip().upper()
    m = re.search(r"\b([ABCDE])\b", t)
    return m.group(1) if m else ""


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
            sleep_s = base_backoff * (2 ** attempt) + random.uniform(0, 0.4)
            time.sleep(sleep_s)
    return "", last_err


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


def build_summary(stats, total_samples, processed_samples, model, run_tag, json_path, base_url, workers):
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
        "base_url": base_url,
        "dataset": str(json_path),
        "run_tag": run_tag,
        "workers": workers,
        "total_samples_in_dataset": total_samples,
        "processed_samples_this_run": processed_samples,
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
    save_raw: bool,
    save_image_abs: bool,
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
    records = []

    for i, pair in enumerate(vqa_pairs):
        qtype = pair.get("question_type", "unknown")
        gt = (pair.get("answer") or "").strip().upper()
        options = pair.get("options", {}) or {}
        is_mcq = "E" in options
        is_trap = bool(pair.get("meta_is_unanswerable", False))

        prompt = build_single_question_prompt(pair, q_index=i + 1)
        messages = [
            {"role": "system", "content": "You are an expert radiologist."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            },
        ]

        partial["api_calls"] += 1
        raw, err = call_api_with_retry(
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
            raw = f"API_ERROR: {err}"

        pred = parse_single_answer(raw)
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

        rec = {
            "id": sample.get("id", si),
            "q_index": i + 1,
            "type": qtype,
            "is_mcq": is_mcq,
            "is_trap": is_trap,
            "gt": gt,
            "pred": pred,
            "correct": ok,
            "image_path": sample.get("image_path", ""),
        }

        if save_raw:
            rec["raw"] = raw

        if save_image_abs:
            # WARNING: may leak local file structure. Disabled by default.
            rec["image_abs"] = str(img_abs)

        records.append(rec)

    return records, partial


def summarize_from_jsonl(jsonl_path: Path, allowed_image_paths: set):
    """
    Summarize only records whose image_path is in allowed_image_paths
    (avoids mixing with old appended runs).
    """
    stats2 = {
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
        "empty_pred": 0,
    }
    by_type2 = defaultdict(lambda: {"total": 0, "correct": 0, "empty_pred": 0})

    parsed_lines = 0
    bad_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed_lines += 1
            try:
                o = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            ip = (o.get("image_path") or "").strip()
            if allowed_image_paths and ip not in allowed_image_paths:
                continue

            t = (o.get("type") or "unknown").strip().lower() or "unknown"
            ok = bool(o.get("correct", False))
            is_mcq = bool(o.get("is_mcq", False))
            is_trap = bool(o.get("is_trap", False))
            pred = (o.get("pred") or "").strip()

            stats2["total"] += 1
            stats2["correct"] += int(ok)

            if pred == "":
                stats2["empty_pred"] += 1
                by_type2[t]["empty_pred"] += 1

            by_type2[t]["total"] += 1
            by_type2[t]["correct"] += int(ok)

            if is_mcq:
                stats2["mcq_total"] += 1
                stats2["mcq_correct"] += int(ok)
            else:
                stats2["binary_total"] += 1
                stats2["binary_correct"] += int(ok)

            if is_trap:
                stats2["trap_total"] += 1
                stats2["trap_correct"] += int(ok)
            else:
                stats2["non_trap_total"] += 1
                stats2["non_trap_correct"] += int(ok)

    def sd(a, b):
        return a / b if b else 0.0

    rows = []
    for t, s in by_type2.items():
        rows.append(
            {
                "type": t,
                "acc": sd(s["correct"], s["total"]),
                "correct": s["correct"],
                "total": s["total"],
                "empty_pred": s["empty_pred"],
            }
        )
    rows.sort(key=lambda r: r["acc"], reverse=True)

    print("\n" + "═" * 70)
    print("📌 STATS FROM JSONL (THIS RUN ONLY)")
    print(f"📄 JSONL: {jsonl_path}")
    print(f"✅ Parsed lines: {parsed_lines - bad_lines}/{parsed_lines} | Bad lines: {bad_lines}")
    print(f"🧪 Samples (unique image_path): {len(allowed_image_paths)}")
    print("═" * 70)
    print(f"🏆 Overall accuracy: {sd(stats2['correct'], stats2['total']):.2%} ({stats2['correct']}/{stats2['total']})")
    print(f"📌 Binary accuracy : {sd(stats2['binary_correct'], stats2['binary_total']):.2%} ({stats2['binary_correct']}/{stats2['binary_total']})")
    print(f"📌 MCQ accuracy    : {sd(stats2['mcq_correct'], stats2['mcq_total']):.2%} ({stats2['mcq_correct']}/{stats2['mcq_total']})")
    print(f"📌 Trap accuracy   : {sd(stats2['trap_correct'], stats2['trap_total']):.2%} ({stats2['trap_correct']}/{stats2['trap_total']})")
    print(f"📌 Non-trap acc    : {sd(stats2['non_trap_correct'], stats2['non_trap_total']):.2%} ({stats2['non_trap_correct']}/{stats2['non_trap_total']})")
    print(f"🕳️ Empty pred      : {stats2['empty_pred']}/{stats2['total']}")
    print("═" * 70)
    print("📊 Accuracy by question_type:")
    for r in rows:
        ep = f", empty_pred={r['empty_pred']}" if r["empty_pred"] else ""
        print(f"  - {r['type']:20s} {r['acc']:.2%} ({r['correct']}/{r['total']}){ep}")
    print("═" * 70)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MM-NeuroOnco closed-ended VQA (single-question-per-call).")
    p.add_argument("--json_path", type=str, required=True, help="Path to closed-ended benchmark JSON.")
    p.add_argument("--data_root", type=str, required=True, help="Root directory containing images.")
    p.add_argument("--extra_roots", type=str, nargs="*", default=[], help="Optional extra roots to search images.")

    p.add_argument("--model", type=str, default=os.environ.get("MMNO_MODEL", "gpt-4.1-mini"))
    p.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", help="Env var for API key.")

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=8)
    p.add_argument("--base_backoff", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=128)

    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--test_n", type=int, default=None, help="Run only first N remaining samples (after resume).")

    p.add_argument("--save_raw", action="store_true", help="Save raw model output to JSONL (large).")
    p.add_argument("--save_image_abs", action="store_true", help="Save absolute image path to JSONL (may leak paths).")

    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key: set environment variable {args.api_key_env} before running.")

    json_path = Path(args.json_path)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    run_tag = args.run_tag or f"{model_name}_closed_singleq"

    out_jsonl = out_dir / f"preds_{run_tag}.jsonl"
    out_summary = out_dir / f"summary_{run_tag}.json"
    out_config = out_dir / f"config_{run_tag}.json"

    # Save config for reproducibility
    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Thread-local client
    thread_local = threading.local()

    def get_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = OpenAI(api_key=api_key, base_url=args.base_url)
        return thread_local.client

    data = json.load(open(json_path, "r", encoding="utf-8"))
    total_samples = len(data)

    done_keys = load_done_keys(out_jsonl)
    print(f"✅ Resume enabled. Found done samples: {len(done_keys)} (by image_path)")

    todo = []
    for si, sample in enumerate(data, start=1):
        ip = sample.get("image_path", "")
        if ip and ip in done_keys:
            continue
        todo.append((si, sample))

    print(f"📦 Total samples in dataset: {total_samples}")
    print(f"🧩 To run (remaining): {len(todo)}")

    if args.test_n is not None:
        todo = todo[: args.test_n]
    print(f"🧪 This run samples: {len(todo)} (test_n={args.test_n})")

    # Track image_path for "this run only" stats
    this_run_image_paths = {s.get("image_path", "") for _, s in todo if s.get("image_path", "")}

    print(f"🌐 base_url={args.base_url}")
    print(f"🤖 model={model_name}")
    print(f"⚡ workers={args.workers}")
    print(f"🧾 OUT_JSONL={out_jsonl}")
    print(f"🧾 OUT_SUMMARY={out_summary}")

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

    write_lock = threading.Lock()
    processed_samples = 0

    # Append for resume
    with open(out_jsonl, "a", encoding="utf-8") as w:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            for si, sample in todo:
                futures.append(
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
                        args.save_raw,
                        args.save_image_abs,
                    )
                )

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Concurrent samples"):
                records, partial = fut.result()

                if records:
                    with write_lock:
                        for r in records:
                            w.write(json.dumps(r, ensure_ascii=False) + "\n")
                        w.flush()

                merge_partial(stats, partial)
                processed_samples += 1

                if processed_samples % args.save_every == 0:
                    summary = build_summary(
                        stats, total_samples, processed_samples, model_name, run_tag, json_path, args.base_url, args.workers
                    )
                    with open(out_summary, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary = build_summary(stats, total_samples, processed_samples, model_name, run_tag, json_path, args.base_url, args.workers)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 60)
    print(f"📊 {model_name} SINGLE-Q (PER QUESTION CALL) + CONCURRENT + RESUME")
    print(f"Newly processed samples: {processed_samples}")
    print(f"API calls: {stats['api_calls']} | API failed: {stats['api_failed']}")
    print(f"🏆 (THIS RUN) OVERALL ACCURACY: {summary['overall_accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print("═" * 60)
    print(f"💾 Summary saved to: {out_summary}")
    print(f"💾 Predictions appended to: {out_jsonl}")

    summarize_from_jsonl(out_jsonl, this_run_image_paths)


if __name__ == "__main__":
    main()
