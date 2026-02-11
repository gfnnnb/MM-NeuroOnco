MM-NeuroOnco Evaluation Guide

This document describes how to set up the environment and run evaluation on the MM-NeuroOnco benchmark.

1️⃣ Environment Setup

Create environment:

conda create -n mm-neuroonco python=3.10 -y
conda activate mm-neuroonco


Install dependencies:

pip install torch>=2.1
pip install transformers>=4.45
pip install accelerate>=0.33
pip install pillow>=10.0
pip install tqdm>=4.66
pip install numpy>=1.24
pip install sentencepiece>=0.2.0
pip install protobuf>=4.25
pip install safetensors>=0.4
pip install einops>=0.8


⚠ Install a PyTorch version compatible with your CUDA version.

2️⃣ Dataset Structure (Important)

The benchmark JSON contains an image_path field.

Example:

{
  "id": 3,
  "image_path": "glioma/16_t1_t2_t1ce_flair_mask/BraTS2021_01648/BraTS2021_01648_flair.png"
}


Your directory structure must satisfy:

data_root / image_path


For example:

/path/to/MM-NeuroOnco-Images/
├── glioma/
│   └── 16_t1_t2_t1ce_flair_mask/
│       └── BraTS2021_01648/
│           └── BraTS2021_01648_flair.png


Then you should pass:

--data_root /path/to/MM-NeuroOnco-Images


The script will internally resolve:

image_abs_path = data_root / image_path

BraTS Images

BraTS2021 images are NOT redistributed due to original dataset licensing.

Please download BraTS2021 separately from:

https://www.kaggle.com/datasets/asefjamilajwad2/brats2021

Place them into the corresponding folder structure matching the JSON file.

3️⃣ Running Evaluation

Example:

CUDA_VISIBLE_DEVICES=0 python eval_script.py \
  --model_dir /path/to/your_model \
  --json_path benchmark/closed/Benchmark_VQA_Closed.json \
  --data_root /path/to/MM-NeuroOnco-Images \
  --out_dir outputs \
  --dtype bfloat16 \
  --resume

4️⃣ Output Files

After evaluation:

outputs/
  preds_<run_tag>.jsonl
  summary_<run_tag>.json


Each JSONL line corresponds to one question:

{
  "id": 1,
  "q_index": 1,
  "type": "diagnosis",
  "is_mcq": true,
  "gt": "C",
  "pred": "C",
  "correct": true,
  "image_path": "1.jpg"
}

5️⃣ Option-E Setting

Some questions include:

E: None of the above


The evaluation scripts automatically detect whether option E exists and adjust the valid answer space accordingly.

6️⃣ Common Errors
Missing Images

If you see missing image errors:

Check that --data_root is correct.

Ensure directory names exactly match the JSON.

Confirm file extensions (png/jpg) match.

7️⃣ Reproducibility Notes

Use --resume to continue interrupted runs.

Use --max_images for quick testing.

Use --dtype float16 if memory is limited.

Avoid saving absolute local paths in public result files.

8️⃣ License

Evaluation scripts: research use.

Benchmark images: hosted on HuggingFace with gated access.

BraTS images: must be downloaded separately.

Users must comply with original dataset licenses.

9️⃣ Citation

If you use this benchmark, please cite:

MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis.