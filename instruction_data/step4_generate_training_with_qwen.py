#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate VQA MCQs from MRI metadata via an LLM endpoint.

Open-source friendly changes:
- No hard-coded API key or private base_url
- Uses env vars + CLI args
- Supports OpenAI-compatible endpoints (OpenAI / vLLM / etc.)
- Removes any private absolute path prefix from image_path (optional)
- Robust resume via id-indexed merge + atomic save
"""

import argparse
import asyncio
import json
import os
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


SYSTEM_PROMPT = r"""
You are a Board-Certified Neuroradiologist. Your task is to transform MRI metadata into professional, VARIED, and IMAGE-BASED MCQs.

### 1. Style & Diversity Guidance (CRITICAL):
* **DIVERSITY**: Do NOT use the exact same sentence template for every case. Vary your phrasing naturally.
  - *Example Diagnosis*: "What is the primary diagnosis?" / "Identify the pathology shown." / "Based on the findings, the most likely diagnosis is:"
  - *Example Features*: "How are the margins described?" / "Describe the observed internal texture." / "The enhancement pattern is best characterized as:"
* **NO CLINICAL HISTORY**: Strictly FORBIDDEN to invent patient age, gender, symptoms, or medical history.
* **BRIEF & VISUAL**: Keep questions concise (under 20 words). Focus only on imaging features.
* **STRICT SOURCE**: Use ONLY provided metadata. IGNORE 'additional_clues'.

### 2. Case Logic (Benchmark Consistent):
* **CASE A: Healthy / Normal (lesion_found is False)**:
    * Generate a **Binary (2-Option) Question**.
    * Question: "Is there a pathological lesion present in this MRI?" (or similar phrasing).
    * Options: {"A": "Tumor / Abnormal", "B": "Healthy / Normal"}.
    * Answer: "B".
* **CASE B: Generic Tumor** (e.g., "Tumor", "Mass"):
    * Generate a **Binary (2-Option) Question**.
    * Options: {"A": "Tumor / Abnormal", "B": "Healthy / Normal"}.
    * Answer: "A".
* **CASE C: Specific Pathology** (e.g., "Meningioma", "Glioma"):
    * Generate a **4-Option MCQ**.
    * Distractors: Must be specific pathologies (e.g., Glioblastoma, Metastasis).

### 3. Feature Logic (Strict Exclusion Rule):
* **IF Healthy / Normal**:
    - **STOP HERE**. Only generate the Diagnosis and Modality MCQs.
* **IF Pathological (Tumor)**:
    - Generate a standard **4-Option MCQ** for EACH provided field:
      (modality, location, shape, margins, texture, signal_intensity, enhancement, edema).

### Output JSON Format (Strict Structure):
{
  "questions": [
    {
      "question_type": "diagnosis",
      "question": "varied_brief_question_text",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "A"
    }
  ]
}
"""


INVALID_VALS = {"null", "none", "nan", "not specified", "unknown", "not assessable", ""}


def clean_json_string(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=os.path.dirname(path) or ".") as tf:
        json.dump(obj, tf, indent=2, ensure_ascii=False)
        tmp_name = tf.name
    os.replace(tmp_name, path)


def load_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input must be a list of items, got: {type(data)}")
    return data


def load_existing(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load existing results and index by id (string).
    This avoids duplicates and supports resume safely.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}
        idx = {}
        for item in data:
            if "id" in item:
                idx[str(item["id"])] = item
        return idx
    except Exception:
        return {}


def normalize_image_path(p: str, strip_prefix: Optional[str]) -> str:
    if not isinstance(p, str):
        return p
    if strip_prefix and p.startswith(strip_prefix):
        return p[len(strip_prefix):]
    return p


def filter_label(label: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (label or {}).items():
        if k == "additional_clues":
            continue
        if v is None:
            continue
        sv = str(v).strip()
        if sv.lower() in INVALID_VALS:
            continue
        out[k] = v
    return out


def is_valid_questions(q: Any) -> bool:
    return isinstance(q, list)


async def call_llm_json(
    client: AsyncOpenAI,
    model: str,
    user_content: str,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        timeout=timeout,
    )
    txt = clean_json_string(resp.choices[0].message.content)
    return json.loads(txt)


async def process_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    item: Dict[str, Any],
    strip_prefix: Optional[str],
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    async with sem:
        item_id = str(item["id"])
        label = filter_label(item.get("label", {}))
        user_content = (
            f"Target Metadata (ID: {item_id}):\n{json.dumps(label, ensure_ascii=False)}\n"
            "Generate professional MCQs with varied phrasing. Image-focused only. No patient history."
        )

        last_err = None
        for attempt in range(max_retries):
            try:
                parsed = await call_llm_json(
                    client=client,
                    model=model,
                    user_content=user_content,
                    temperature=temperature,
                    timeout=timeout,
                )
                questions = parsed.get("questions", [])
                if not is_valid_questions(questions):
                    raise ValueError(f"Invalid 'questions' format: {type(questions)}")

                return {
                    "id": item_id,
                    "image_path": normalize_image_path(item.get("image_path", ""), strip_prefix),
                    "vqa_pairs": questions,
                }
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                continue

        return {
            "id": item_id,
            "image_path": normalize_image_path(item.get("image_path", ""), strip_prefix),
            "vqa_pairs": [],
            "error": str(last_err),
        }


async def main_async(args: argparse.Namespace) -> None:
    # Prefer env vars, allow CLI override
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", None)

    if not api_key:
        # For many self-hosted endpoints, a dummy key is acceptable.
        # Keep it explicit so users know how to set it.
        api_key = "EMPTY"

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=1,
    )

    all_input = load_input(args.input)
    existing_idx = load_existing(args.output)

    # Skip only successful ones (no error and non-empty vqa_pairs)
    def is_success(x: Dict[str, Any]) -> bool:
        return (not x.get("error")) and bool(x.get("vqa_pairs"))

    processed_success = {k for k, v in existing_idx.items() if is_success(v)}

    tasks_data = [x for x in all_input if str(x.get("id")) not in processed_success]
    if args.run_limit is not None:
        tasks_data = tasks_data[: args.run_limit]

    print(f"📊 Pending: {len(tasks_data)} / Total: {len(all_input)}")
    if not tasks_data:
        print("✅ Nothing to do (all successful samples already exist).")
        return

    sem = asyncio.Semaphore(args.concurrency)
    save_lock = asyncio.Lock()

    # We'll store results in an id-indexed dict (merge-safe)
    results_idx = dict(existing_idx)

    pbar = tqdm_asyncio(total=len(tasks_data), desc="Generating (Async)")

    async def runner(one_item: Dict[str, Any]):
        nonlocal results_idx
        out = await process_one(
            sem=sem,
            client=client,
            model=args.model,
            item=one_item,
            strip_prefix=args.strip_prefix,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        results_idx[str(out["id"])] = out
        pbar.update(1)

        async with save_lock:
            if (pbar.n % args.save_interval) == 0:
                atomic_json_dump(list(results_idx.values()), args.output)

    await asyncio.gather(*[runner(x) for x in tasks_data])

    atomic_json_dump(list(results_idx.values()), args.output)

    # Summary
    all_results = list(results_idx.values())
    num_failed = sum(1 for x in all_results if x.get("error"))
    num_success = sum(1 for x in all_results if (not x.get("error")) and x.get("vqa_pairs"))
    num_pairs = sum(len(x.get("vqa_pairs", [])) for x in all_results)

    print(f"🎉 Done: {args.output}")
    print(f"✅ Success: {num_success}, ❌ Failed: {num_failed}, 🧩 Total QA pairs: {num_pairs}")

    type_counter = Counter()
    for x in all_results:
        for qa in x.get("vqa_pairs", []):
            type_counter[qa.get("question_type", "unknown")] += 1
    if type_counter:
        print("📌 Question Type Distribution (Top 20):")
        for k, v in type_counter.most_common(20):
            print(f"  {k:20s}: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="Train_Dataset_Cleaned.json", help="Input metadata JSON file")
    p.add_argument("--output", default="Final_Train_VQA.json", help="Output VQA JSON file")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen3-Next-80B-A3B-Instruct"))
    p.add_argument("--base_url", default=None, help="OpenAI-compatible base URL (or set OPENAI_BASE_URL)")
    p.add_argument("--api_key", default=None, help="API key (or set OPENAI_API_KEY)")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--run_limit", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.3)

    # This is the "minimum modification" you wanted for open-sourcing:
    # remove your private absolute prefix, keep the rest (including BraTS2021_00352).
    p.add_argument(
        "--strip_prefix",
        default=os.environ.get("STRIP_PATH_PREFIX", "/data/shared/Brain_tumor_dataset/"),
        help="If image_path starts with this prefix, remove it. Set empty to disable.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.strip_prefix == "":
        args.strip_prefix = None
    asyncio.run(main_async(args))
