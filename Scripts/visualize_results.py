# run_finetuned_clip.py

import numpy as np
import os
import pandas as pd
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

# === 1. Setup ===
dataset_root = "/kaggle/input/superstition-dataset/Big Data"  # ✅ Change if needed
# model_path = "fine_tuned_model.pt"  # ✅ Fine-tuned model path
model_path = '/home/aarish/VLM-superstition-analysis/models/superstition_clip_final.pt'
output_base_dir = "fine_tuned_results"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. Load fine-tuned CLIP ===
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model"])
model = model.float().to(device).eval()
print("✅ Loaded fine-tuned CLIP model.")

# === 3. Valid Image Check ===
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            return True
    except:
        return False

# === 4. Load Image Paths ===
def load_dataset(path):
    data = {"path": [], "label": []}
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                if is_valid_image(full_path):
                    label = os.path.basename(root)
                    data["path"].append(full_path)
                    data["label"].append(label)
    return pd.DataFrame(data)

# === 5. Category and Prompts ===
category_terms = {
    "plant_images": "plant",
    "animal_images": "animal",
    "object_images": "object",
    "number_images": "number",
    "color_images": "color",
    "places_images": "place",
    "symbol_images": "symbol",
    "natural_phenomena_images": "natural phenomenon"
}

superstition_signs = [
    "good luck",
    "bad luck",
    "wealth",
    "loss",
    "prosperity",
    "illness"
]

# === 6. Process Each Category ===
for category_folder, base_term in category_terms.items():
    print(f"\n🔍 Processing: {category_folder}")

    category_path = os.path.join(dataset_root, category_folder)
    dataset = load_dataset(category_path)
    image_paths = dataset["path"].tolist()

    if not image_paths:
        print(f"⚠️ No valid images in: {category_folder}")
        continue

    # Preprocess images
    preprocessed_images = []
    valid_image_paths = []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            preprocessed_images.append(image)
            valid_image_paths.append(path)
        except Exception as e:
            print(f"❌ Failed image {path}: {e}")

    if not preprocessed_images:
        continue

    with torch.no_grad():
        image_features = model.encode_image(torch.cat(preprocessed_images)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Create and encode prompts
    prompts = [f"Image of {base_term} which is a sign of {sign}" for sign in superstition_signs]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    temperature = 0.1
    logits = image_features @ text_features.T
    logits /= temperature
    probs = logits.softmax(dim=-1)

    # Save top images per prompt
    def save_top_images(prompt, prompt_index, top_k=10):
        prompt_safe = prompt.replace(" ", "_").replace("/", "_")
        prompt_dir = os.path.join(output_base_dir, category_folder, f"{prompt_index}_{prompt_safe}")
        os.makedirs(prompt_dir, exist_ok=True)

        top_indices = probs[:, prompt_index].topk(top_k).indices
        top_scores = probs[:, prompt_index].topk(top_k).values

        print(f"\n📌 Prompt: '{prompt}' — Top {top_k} matches:")
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            img_path = valid_image_paths[idx.item()]
            print(f"{i+1}. {img_path} — Score: {score.item():.4f}")
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(os.path.join(prompt_dir, f"{i+1}_score_{score.item():.4f}.jpg"))
            except Exception as e:
                print(f"Error saving: {img_path} — {e}")

    for i, prompt in enumerate(prompts):
        save_top_images(prompt, i, top_k=100)

print("\n✅ Evaluation complete using fine-tuned CLIP model.")
