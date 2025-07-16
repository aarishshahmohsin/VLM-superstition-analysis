# run_vit_l14_zeroshot.py

import os
import torch
import clip
import pandas as pd
from PIL import Image

# Path setup
dataset_root = "data/superstition-dataset/Big Data"
output_base_dir = "parsed_results/V14"
os.makedirs(output_base_dir, exist_ok=True)

# Load ViT-L/14 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Define superstition categories
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
    "good luck", "bad luck", "wealth", "loss", "prosperity", "illness"
]

# Helper: Validate image

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            return True
    except:
        return False

# Helper: Load images

def load_dataset(path):
    data = {"path": [], "label": []}
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                if is_valid_image(full_path):
                    label = os.path.basename(root)
                    data["path"].append(full_path)
                    data["label"].append(label)
    return pd.DataFrame(data)

# Main inference loop
for category_folder, base_term in category_terms.items():
    print(f"\n🔍 Evaluating category: {category_folder}")
    category_path = os.path.join(dataset_root, category_folder)
    df = load_dataset(category_path)
    image_paths = df["path"].tolist()

    if not image_paths:
        print(f"⚠️ No images in {category_folder}")
        continue

    images, valid_paths = [], []
    for path in image_paths:
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            images.append(image)
            valid_paths.append(path)
        except:
            continue

    with torch.no_grad():
        image_features = model.encode_image(torch.cat(images)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    prompts = [f"Image of {base_term} which is a sign of {s}" for s in superstition_signs]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1).cpu().numpy()

    # Save top 100 images per prompt
    for i, prompt in enumerate(prompts):
        prompt_safe = prompt.replace(" ", "_")
        topk = probs[:, i].argsort()[-100:][::-1]

        output_path = os.path.join(output_base_dir, f"{category_folder}_{i}_{prompt_safe}.txt")
        with open(output_path, "w") as f:
            f.write(f"📌 Prompt: '{prompt}'\n")
            for rank, idx in enumerate(topk):
                f.write(f"{rank+1}. {valid_paths[idx]} — Score: {probs[idx][i]:.4f}\n")

    print(f"✅ Done with {category_folder}")
