# Metadata Extraction Module

This folder contains scripts used to clean and standardize raw multi-source MRI datasets
during the construction of MM-NeuroOnco.

It is part of the **data preprocessing pipeline** and is not required for benchmark evaluation.

---

## What This Module Does

- Traverse raw dataset directories
- Parse YOLO segmentation `.txt` annotations
- Convert polygon annotations into binary mask images (`_mask.png`)
- Prepare standardized data for downstream metadata extraction

This module is used only during dataset construction.

---

## Path Configuration

To ensure portability and safe open-source release:

- No absolute paths (e.g., `/home/user/...`) are hard-coded.
- Dataset paths must be specified via argument or environment variable.
- A default relative path may be provided inside the script.

### Running the Script

You may provide the dataset root using one of the following methods:

```bash
# Option 1: Command-line argument (recommended)
python yolo_txt_to_mask.py --dataset_root /path/to/raw_dataset

# Option 2: Environment variable
export DATASET_ROOT=/path/to/raw_dataset
python yolo_txt_to_mask.py

# Example with keyword filtering
python yolo_txt_to_mask.py --dataset_root /data/raw --keyword 19
```

## Parameters

| Argument | Description |
|----------|-------------|
| `--dataset_root` | Root directory of raw datasets |
| `--keyword` | Only process subfolders containing this keyword |

---

## Notes

- This module contains no private data.
- No API keys or credentials are stored.
- All paths are OS-independent.
- Output is deterministic.

---

## Role in MM-NeuroOnco

Raw datasets  
→ Mask generation  
→ Metadata extraction  
→ Benchmark & Instruction Dataset construction  
