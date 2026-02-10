import os
import json
import base64
import imghdr
import time
import random
import threading
import argparse
import hashlib
from typing import Any, Dict, Optional

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== PROMPT ====================
SYSTEM_PROMPT = r"""<YOUR SYSTEM PROMPT HERE, UNCHANGED>"""
# ↑ 你把原 SYSTEM_PROMPT 整段原样粘贴进来（我这里不重复贴，避免你复制时漏行）


CORE_FIELDS = ["shape", "margins", "texture", "enhancement", "edema", "signal_intensity"]


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 1e-6)
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed_time:
                time.sleep(self.next_allowed_time - now)
            self.next_allowed_time = time.time() + self.min_interval


print_lock = threading.Lock()


def safe_tqdm_write(msg: str):
    with print_lock:
        tqdm.write(msg)


def is_null(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"null", "none", "unknown", ""}


def count_filled_6(signs: Any) -> int:
    if not isinstance(signs, dict):
        return 0
    return sum(1 for k in CORE_FIELDS if k in signs and not is_null(signs.get(k)))


def normalize_prediction_obj(res: Any) -> Dict[str, Any]:
    default_meta = {"predicted_modality": "Unknown", "predicted_location": "null", "lesion_found": "False"}
    default_signs = {k: "null" for k in CORE_FIELDS}
    if not isinstance(res, dict):
        return {"meta_info": default_meta, "structured_signs": default_signs}
    meta = res.get("meta_info")
    signs = res.get("structured_signs")
    if not isinstance(meta, dict):
        meta = default_meta
    if not isinstance(signs, dict):
        signs = default_signs
    for k in CORE_FIELDS:
        if k not in signs:
            signs[k] = "null"
    res["meta_info"] = meta
    res["structured_signs"] = signs
    return res


def detect_mime_from_bytes(b: bytes, path: str = "") -> str:
    kind = imghdr.what(None, h=b)
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def extract_content_text(choice_message_content: Any) -> Optional[str]:
    if choice_message_content is None:
        return None
    if isinstance(choice_message_content, str):
        return choice_message_content
    if isinstance(choice_message_content, list):
        parts = [x.get("text", "") if isinstance(x, dict) else str(x) for x in choice_message_content]
        return "\n".join(parts).strip() or None
    return str(choice_message_content).strip() or None


def stable_short_id(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def call_llm_with_retry(client: OpenAI, rate_limiter: RateLimiter, model: str, temperature: float, messages):
    max_retries = 6
    base_backoff = 1.2
    for attempt in range(max_retries):
        try:
            rate_limiter.acquire()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = extract_content_text(resp.choices[0].message.content)
            if not content:
                raise RuntimeError("Empty LLM response")
            return content
        except Exception as e:
            msg = str(e).lower()
            retryable = ("429" in msg) or ("rate limit" in msg) or ("timeout" in msg) or ("overloaded" in msg)
            if not retryable:
                raise
            time.sleep(min(30.0, base_backoff * (2 ** attempt)) + random.uniform(0.0, 0.8))
    raise RuntimeError("Max retries exceeded")


def save_checkpoint(data_list, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    temp_path = filepath + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, filepath)


def print_batch_report(data_list):
    if not data_list:
        return
    print("\n" + "=" * 60)
    print(f"📊 Cumulative Extraction Report (Total Processed: {len(data_list)})")
    print("=" * 60)

    counts = {k: 0 for k in CORE_FIELDS}
    valid_samples = 0

    for item in data_list:
        pred = item.get("ai_prediction", {})
        meta = pred.get("meta_info", {})
        if str(meta.get("lesion_found")) == "True":
            valid_samples += 1
            signs = pred.get("structured_signs", {})
            for k in CORE_FIELDS:
                if not is_null(signs.get(k)):
                    counts[k] += 1

    if valid_samples == 0:
        print("⚠️ No lesion-positive samples found; skip stats.")
        return

    field_rates = []
    for k in CORE_FIELDS:
        rate = (counts[k] / valid_samples) * 100
        field_rates.append(rate)
        bar_len = int(rate / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{k.ljust(17)}: {bar} {rate:5.1f}%")

    avg_air = sum(field_rates) / len(CORE_FIELDS)
    print("-" * 60)
    print(f"🎯 Based on {valid_samples} lesion-positive samples")
    print(f"🌟 Avg extraction rate: {avg_air:5.2f}%")
    print("=" * 60 + "\n")


def process_single_item(
    client: OpenAI,
    rate_limiter: RateLimiter,
    dataset_root: str,
    model: str,
    temperature: float,
    item: Dict[str, Any],
    log_sample_id: bool,
):
    sample_id = item.get("sample_id", "") or ""
    sid_show = sample_id if log_sample_id else stable_short_id(sample_id) if sample_id else "noid"

    try:
        img_rel_path = (item.get("image_path") or "").replace("\\", "/")
        img_full_path = os.path.join(dataset_root, img_rel_path)

        if not img_rel_path or not os.path.exists(img_full_path):
            raise FileNotFoundError(f"image_missing: {img_rel_path}")

        with open(img_full_path, "rb") as f:
            img_bytes = f.read()
        mime = detect_mime_from_bytes(img_bytes, img_full_path)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Context: {item.get('coarse_description', '')}"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                ],
            },
        ]

        content = call_llm_with_retry(client, rate_limiter, model, temperature, messages)
        res = normalize_prediction_obj(json.loads(content))

        signs = res.get("structured_signs", {})
        filled = count_filled_6(signs)
        rate = (filled / 6) * 100
        meta = res.get("meta_info", {})
        icon = "🟢" if str(meta.get("lesion_found")) == "True" else "⚪"
        loc = meta.get("predicted_location")

        # keep your rule: lesion_found true but loc missing -> set false
        if icon == "🟢" and (loc is None or str(loc).strip() == "" or str(loc).lower() == "null"):
            meta["lesion_found"] = "False"
            icon = "⚪"
            res["meta_info"] = meta

        safe_tqdm_write(f"{icon} [{sid_show}] Loc: {str(loc):<12} | Filled: {filled}/6 ({rate:>3.0f}%)")
        return {"sample_id": sample_id, "ai_prediction": res}

    except Exception as e:
        safe_tqdm_write(f"🚨 Error [{sid_show}]: {e}")
        return {"sample_id": sample_id, "ai_prediction": {"error": str(e)}}


def main():
    parser = argparse.ArgumentParser(description="LLM-based structured sign extraction (open-source safe).")
    parser.add_argument("--run_name", type=str, default="run_extraction", help="Run name for output naming.")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model name.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--rps", type=float, default=1.5, help="Client-side rate limit (requests/sec).")

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_ROOT", "./data"),
        help="Dataset root directory that contains images. Default: $DATASET_ROOT or ./data",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("INPUT_FILE", "./outputs/step1_input_balanced.json"),
        help="Input JSON with sample_id/image_path/coarse_description. Default: $INPUT_FILE or ./outputs/step1_input_balanced.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT_FILE", "./outputs/silver_label_extract.json"),
        help="Output JSON (checkpoint + final). Default: $OUTPUT_FILE or ./outputs/silver_label_extract.json",
    )
    parser.add_argument(
        "--gold_cache",
        type=str,
        default=os.getenv("FINAL_GOLD_FILE", "./outputs/Final_Gold_Standard_CLEAN.json"),
        help="Optional: existing final gold JSON to skip processed ids. Default: $FINAL_GOLD_FILE or ./outputs/Final_Gold_Standard_CLEAN.json",
    )
    parser.add_argument(
        "--log_sample_id",
        action="store_true",
        help="Log raw sample_id (default logs a short hash to reduce leakage).",
    )
    args = parser.parse_args()

    # --- Secrets from env ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Refusing to run to avoid hardcoded secrets.")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    rate_limiter = RateLimiter(args.rps)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    if not isinstance(all_data, list):
        raise ValueError("Input JSON must be a list of items.")

    processed_ids = set()
    completed_results = []

    # resume from output
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                completed_results = json.load(f)
            processed_ids.update({it.get("sample_id") for it in completed_results if it.get("sample_id")})
            print(f"♻️ Resume: loaded {len(completed_results)} existing outputs -> skipping them")
        except Exception:
            pass

    # skip ids from gold cache
    if args.gold_cache and os.path.exists(args.gold_cache):
        try:
            with open(args.gold_cache, "r", encoding="utf-8") as f:
                gold_data = json.load(f)
            gold_ids = {it.get("sample_id") for it in gold_data if it.get("sample_id")}
            processed_ids.update(gold_ids)
            print(f"🏆 Gold cache: {len(gold_ids)} ids loaded -> skipping them")
        except Exception:
            pass

    pending_items = [it for it in all_data if it.get("sample_id") not in processed_ids]
    if not pending_items:
        print("🎉 Nothing to do. All items already processed.")
        return

    current_batch = pending_items[: args.batch_size]
    print(f"🚀 Run={args.run_name} | batch={len(current_batch)} | remaining={len(pending_items)}")
    print("=" * 60)

    processed_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_single_item,
                client,
                rate_limiter,
                args.dataset_root,
                args.model,
                args.temperature,
                item,
                args.log_sample_id,
            )
            for item in current_batch
        ]
        for fu in tqdm(as_completed(futures), total=len(futures)):
            res = fu.result()
            completed_results.append(res)
            processed_count += 1
            if processed_count % args.save_interval == 0:
                save_checkpoint(completed_results, args.output)

    save_checkpoint(completed_results, args.output)
    print(f"\n✅ Finished {processed_count} items. Saved to: {args.output}")
    print_batch_report(completed_results)


if __name__ == "__main__":
    main()
