import os
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset root
dataset_root = "/kaggle/input/superstition-dataset/Big Data"

# Category to term
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

# Superstition and opposites
superstition_signs = ["good luck", "bad luck", "wealth", "loss", "prosperity", "illness"]
opposite_signs = {
    "good luck": "bad luck",
    "bad luck": "good luck",
    "wealth": "loss",
    "loss": "wealth",
    "prosperity": "illness",
    "illness": "prosperity"
}

# Output
output_csv = "clip_superstition_dataset_second_half.csv"
data = []

# For each category
for category_folder, base_term in category_terms.items():
    category_path = os.path.join(dataset_root, category_folder)
    if not os.path.exists(category_path):
        continue

    print(f"\n📁 Processing category: {category_folder}")

    # Collect paths grouped by label
    label_image_paths = {}
    for subdir, _, files in os.walk(category_path):
        label = os.path.basename(subdir)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths = [os.path.join(subdir, f) for f in image_files]
        if image_paths:
            label_image_paths[label] = image_paths

    # Process only the second half of each label
    for label, paths in label_image_paths.items():
        paths.sort()  # Ensure consistent ordering
        half = len(paths) // 2
        second_half_paths = paths[half:]

        for image_path in tqdm(second_half_paths, desc=f"{category_folder} → {label}"):
            try:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

                # Create prompts
                prompts = [f"Image of a {base_term} which is a sign of {s}" for s in superstition_signs]
                text = clip.tokenize(prompts).to(device)

                # CLIP similarity
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    logits_per_image = image_features @ text_features.T
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

                # Top superstition match
                top_idx = int(probs.argmax())
                stereotype = superstition_signs[top_idx]
                counter = opposite_signs[stereotype]

                # Final prompts
                neutral_prompt = f"Image of a {base_term}"
                stereotype_prompt = prompts[top_idx]
                counter_prompt = f"Image of a {base_term} which is a sign of {counter}"

                data.append({
                    "image_path": image_path,
                    "label": label,
                    "category": category_folder,
                    "neutral_prompt": neutral_prompt,
                    "stereotype_prompt": stereotype_prompt,
                    "counter_prompt": counter_prompt,
                    "predicted_label": stereotype
                })

            except Exception as e:
                print(f"⚠️ Failed on {image_path}: {e}")

# Save
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"\n✅ Saved: {output_csv}")
print(f"🖼️ Total samples (second half only): {len(df)}")
