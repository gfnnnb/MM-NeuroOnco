🧠 Brain Tumor Multimodal Label Construction Pipeline
Overview

This repository provides the full data construction pipeline for:

📦 Multi-source dataset aggregation

🧬 Automated medical semantic extraction

🤖 Multi-model silver label generation

🧪 High-precision dual-model fusion

🩺 Conservative radiology-level quality control

🏆 Final gold-standard label construction

This pipeline was designed for clinically grounded multimodal brain tumor reasoning research.

📂 Project Structure
.
├── metadata_extraction/
│   ├── process_dataset_05.py
│   ├── process_dataset_10.py
│   ├── process_dataset_14.py
│   ├── process_dataset_16.py
│   ├── process_dataset_19.py
│   └── merge_metadata.py
│
├── pipeline/
│   ├── prepare_llm_inference_json.py
│   ├── extract_semantic_features.py
│   ├── run_a_label_extraction.py
│   ├── run_b_label_extraction.py
│   ├── fuse_silver_labels.py
│   └── step3_quality_control.py
│
├── outputs/
├── requirements.txt
└── README.md

⚙️ Installation
git clone <your_repo_url>
cd <repo_name>
pip install -r requirements.txt

🔐 API Key Setup

This project requires an LLM API.

Set your key as an environment variable:

export OPENAI_API_KEY="your_api_key_here"


(Optional, if using custom endpoint)

export OPENAI_BASE_URL="https://your-endpoint"


No API keys are stored in the repository.

🏗 Pipeline Workflow
Step 1 — Metadata Extraction

Generate unified metadata from raw datasets:

python metadata_extraction/process_dataset_05.py
python metadata_extraction/process_dataset_10.py
...


Then merge:

python metadata_extraction/merge_metadata.py

Step 2 — Medical Semantic Extraction
python pipeline/extract_semantic_features.py \
  --input ./outputs/all_brain_tumor_metadata.json \
  --output ./outputs/all_brain_tumor_metadata_rich.json

Step 3 — Coarse Description Generation
python pipeline/prepare_llm_inference_json.py \
  --input ./outputs/all_brain_tumor_metadata_rich.json \
  --output ./outputs/Dataset_For_LLM_Inference.json

Step 4 — Silver Label Generation (Dual Models)

Run A:

python pipeline/run_a_label_extraction.py \
  --dataset_root ./data \
  --input ./outputs/Dataset_For_LLM_Inference.json


Run B:

python pipeline/run_b_label_extraction.py \
  --dataset_root ./data \
  --input ./outputs/Dataset_For_LLM_Inference.json

Step 5 — High-Precision Fusion
python pipeline/fuse_silver_labels.py \
  --file_a ./outputs/silver_label_extract_run_a.json \
  --file_b ./outputs/silver_label_extract_run_b.json

Step 6 — Final Gold Standard QC
python pipeline/step3_quality_control.py \
  --dataset_root ./data \
  --input ./outputs/Fused_Silver_Labels_HIGH_PRECISION.json

🧪 Output Files
File	Description
all_brain_tumor_metadata.json	Unified metadata
all_brain_tumor_metadata_rich.json	Metadata + semantic features
Dataset_For_LLM_Inference.json	LLM-ready dataset
silver_label_extract_run_a.json	Silver labels (Model A)
silver_label_extract_run_b.json	Silver labels (Model B)
Fused_Silver_Labels_HIGH_PRECISION.json	Consensus tumor labels
Final_Gold_Standard_CLEAN.json	Final audited gold labels
🧠 Design Philosophy

This pipeline emphasizes:

Conservative clinical reasoning

Dual-model cross-validation

Physics-aware modality auditing

Explicit hallucination control

Structured sign consistency enforcement

📌 Notes

This repository contains code only, no dataset.

Users must provide their own MRI dataset under --dataset_root.

All outputs are reproducible with identical seeds and temperature=0.


🔒 Reproducibility

Deterministic decoding (temperature=0)

Atomic file writes

Checkpoint-based recovery

Multi-thread safe execution