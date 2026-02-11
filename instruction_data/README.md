instruction_data/README.md
Instruction Tuning Data

This directory contains instruction-format training datasets used for supervised fine-tuning (SFT) of multimodal vision-language models in MM-NeuroOnco.

Due to file size limitations, the full instruction datasets are hosted on Hugging Face:

👉 https://huggingface.co/datasets/gfnnnb/MM-NeuroOnco-Images

Each file is provided in JSONL format, where each line is an independent training sample.

📂 Available Files

instruction_open_train.jsonl

instruction_closed_cot_train.jsonl

instruction_closed_nocot_train.jsonl

📄 Data Format

Each line follows a chat-style multimodal structure:

{
  "images": ["relative/path/to/image.jpg"],
  "messages": [
    {
      "role": "user",
      "content": "<image>\nQuestion text..."
    },
    {
      "role": "assistant",
      "content": "Model response text..."
    }
  ]
}


Some datasets (e.g., open-ended) also contain an auxiliary meta field:

"meta": {
  "id": 1,
  "pair_index": 0,
  "question_type": "analysis",
  "subtype": "global",
  "source": "Train_Dataset_Open_question"
}


The meta field is preserved for traceability but is not required for training.

🧠 Dataset Variants
1️⃣ Open-Ended Instructions

Example structure:

{
  "images": ["relative/path/to/image.jpg"],
  "messages": [
    {
      "role": "user",
      "content": "<image>\nQuestion: Describe the key imaging findings..."
    },
    {
      "role": "assistant",
      "content": "Free-text diagnostic description..."
    }
  ],
  "meta": {...}
}


Characteristics:

Assistant outputs full free-text medical analysis

No multiple-choice options

Designed for generative diagnostic reasoning

Includes additional metadata for analysis

2️⃣ Closed-Ended (No CoT)

Example structure:

{
  "images": ["relative/path/to/image.png"],
  "messages": [
    {
      "role": "user",
      "content": "<image>\nIdentify the pathology...\n\nOptions:\nA. ...\nB. ...\n\nAnswer with the option letter only."
    },
    {
      "role": "assistant",
      "content": "D"
    }
  ]
}


Characteristics:

Multiple-choice format

Assistant outputs only a single option letter

No explanation or reasoning text

Used for direct classification supervision

3️⃣ Closed-Ended (With CoT)

Example structure:

{
  "images": ["relative/path/to/image.png"],
  "messages": [
    {
      "role": "user",
      "content": "<image>\nQuestion text...\nOptions:\n..."
    },
    {
      "role": "assistant",
      "content": "Final: D\nExplanation: ..."
    }
  ]
}


Characteristics:

Multiple-choice format

Assistant outputs:

Final answer letter

Brief medical reasoning explanation

Used to enhance structured diagnostic reasoning

🖼 Image Handling

images contains relative paths only.

No absolute server paths are distributed.

No original MRI images are included.

Users must obtain official datasets independently.

⚠ Licensing Notice

This repository does not redistribute original medical imaging datasets (e.g., BraTS).

Only derived instruction annotations and metadata are distributed.

Users are responsible for complying with the licenses of the original datasets.

🔁 Reproducibility

The instruction datasets were generated using:

Structured medical metadata

Multi-model semantic extraction

Controlled prompt templates

Anti-leakage constraints

See the main repository for generation and evaluation scripts.