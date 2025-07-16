import os
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset root directory
dataset_root = "./dataset/Big Data"

# Output CSV file
output_csv = "clip_superstition_dataset.csv"

# Category to base term mapping
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

# Superstition signs and their opposites
superstition_signs = ["good luck", "bad luck", "wealth", "loss", "prosperity", "illness"]
opposite_signs = {
    "good luck": "bad luck",
    "bad luck": "good luck",
    "wealth": "loss",
    "loss": "wealth",
    "prosperity": "illness",
    "illness": "prosperity"
}

data = []

# Iterate over each category
for category_folder, base_term in category_terms.items():
    category_path = os.path.join(dataset_root, category_folder)
    if not os.path.exists(category_path):
        continue

    for subdir, _, files in os.walk(category_path):
        for file in tqdm(files, desc=category_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(subdir, file)

                try:
                    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

                    prompts = [f"Image of a {base_term} which is a sign of {s}" for s in superstition_signs]
                    text_tokens = clip.tokenize(prompts).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image)
                        text_features = model.encode_text(text_tokens)
                        logits_per_image = image_features @ text_features.T
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

                    top_idx = int(probs.argmax())
                    stereotype = superstition_signs[top_idx]
                    counter = opposite_signs[stereotype]

                    data.append({
                        "image_path": image_path,
                        "neutral_prompt": f"Image of a {base_term}",
                        "stereotype_prompt": f"Image of a {base_term} which is a sign of {stereotype}",
                        "counter_prompt": f"Image of a {base_term} which is a sign of {counter}"
                    })

                except Exception as e:
                    print(f"⚠️ Failed on {image_path}: {e}")

# Save dataset
os.makedirs("parsed_results", exist_ok=True)
df = pd.DataFrame(data)
df.to_csv(os.path.join("parsed_results", output_csv), index=False)
print(f"✅ Saved: parsed_results/{output_csv}")
print(f"🖼️ Total samples: {len(df)}")
