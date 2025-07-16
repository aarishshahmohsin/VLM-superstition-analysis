

---

# 🌠 Superstition Detection in VLM using CLIP

This project explores superstition-based biases in vision-language models (VLMs), specifically OpenAI's CLIP. We evaluate how CLIP models like `ViT-B/32` and `ViT-L/14` respond to prompts that associate images with superstitious beliefs (e.g., "Image of a black cat which is a sign of bad luck").

We also fine-tune the CLIP model using contrastive loss on a superstition-labeled dataset and compare results against zero-shot performance.

---

## 📂 Dataset

* Dataset: [Superstition Dataset](https://www.kaggle.com/datasets/anas123siddiqui/superstition-dataset)
* Structure:

  ```
  Big Data/
  ├── animal_images/
  ├── plant_images/
  ├── color_images/
  ├── number_images/
  ├── object_images/
  ├── places_images/
  ├── symbol_images/
  └── natural_phenomena_images/
  ```

Each subfolder contains images related to a specific superstition category.

---

## 🧠 Model Versions

* `ViT-B/32` — OpenAI's CLIP base model
* `ViT-L/14` — Larger CLIP model
* `Fine-Tuned CLIP-B/32` — Trained using contrastive loss with superstition prompts

---

## 🧪 Project Pipeline

### 1. **Zero-Shot Evaluation**

* `run_vit_b32_zeroshot.py`: Run zero-shot on `ViT-B/32`
* `run_vit_l14_zeroshot.py`: Run zero-shot on `ViT-L/14`

### 2. **Dataset Creation for Fine-Tuning**

* `create_finetune_dataset.py`: Generates a CSV with neutral, stereotype, and counter prompts using CLIP probabilities.

### 3. **Fine-Tuning**

* `finetune_clip.py`: Fine-tunes `ViT-B/32` on the created dataset using contrastive loss and K-Fold CV.

### 4. **Evaluation on Fine-Tuned Model**

* `run_finetuned_clip.py`: Runs inference using the fine-tuned model on the dataset and saves ranked outputs.

### 5. **Visualization**

* `visualize_results.py`: Generates comparative bar plots showing top predicted superstition categories across models.

---

## 🔧 Requirements

```bash
pip install torch torchvision ftfy regex tqdm matplotlib pandas seaborn pillow
pip install git+https://github.com/openai/CLIP.git
```

---

## 🧩 Run Everything

To execute all steps in order:

```bash
python main.py
```

---

## 📁 File Structure

```
.
VLM-superstition-analysis/

│
├── scripts/
│   ├── run_vit_b32_zeroshot.py     # ViT-B/32 zero-shot eval
│   ├── run_vit_l14_zeroshot.py     # ViT-L/14 zero-shot eval
│   ├── create_finetune_dataset.py  # Create CSV for fine-tuning
│   ├── finetune_clip.py            # Fine-tune CLIP model
│   ├── run_finetuned_clip.py       # Evaluate fine-tuned model
│   └── visualize_results.py        # Final value counts and visualization
│
├── main.py                         # 🚀 Master script to run all stages
├── README.md                       # 📘 Instructions and explanation
├── requirements.txt                # 🧪 Dependencies
└── results/                 # Stores CSVs from each phase

```

---

## 🔗 Model and Dataset Links

* 🔍 **Fine-tuned Model (ViT-B/32)**:
  [HuggingFace - Finetuned CLIP](https://huggingface.co/Mohammad121/Finetuned_CLIP-32_no_superstition/blob/main/fine_tuned_model.pt)

* 📁 **Dataset**:
  [Kaggle - Superstition Dataset](https://www.kaggle.com/datasets/anas123siddiqui/superstition-dataset)

---

## 📊 Results Summary

* Bar plots show how different models rank categories (e.g., black cat as "bad luck").
* Fine-tuned CLIP model demonstrates reduced stereotypical bias vs. zero-shot.

---

## 🤝 Contributing

Pull requests, suggestions, and new prompt expansions are welcome!

---

## 🧠 Citation

If you use this project for your research, please consider citing or linking back to this repository.

---

Let me know if you'd like me to help zip the full repo for upload or write a `setup.py` or Jupyter notebook version as well.
