
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import clip

# Superstition terms and their opposites
SUPERSTITIONS = ["good luck", "bad luck", "wealth", "loss", "prosperity", "illness"]
OPPOSITES = {
    "good luck": "bad luck",
    "bad luck": "good luck",
    "wealth": "loss",
    "loss": "wealth",
    "prosperity": "illness",
    "illness": "prosperity"
}

CATEGORY_TERMS = {
    "plant_images": "plant",
    "animal_images": "animal",
    "object_images": "object",
    "number_images": "number",
    "color_images": "color",
    "places_images": "place",
    "symbol_images": "symbol",
    "natural_phenomena_images": "natural phenomenon"
}

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def create_clip_prompt_dataset(dataset_root, output_csv='output/clip_superstition_dataset.csv', model_name='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    data = []

    for category_folder, base_term in CATEGORY_TERMS.items():
        category_path = os.path.join(dataset_root, category_folder)
        if not os.path.exists(category_path):
            continue

        for subdir, _, files in os.walk(category_path):
            for file in tqdm(files, desc=category_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subdir, file)
                    if not is_valid_image(image_path):
                        continue

                    try:
                        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                        prompts = [f"Image of a {base_term} which is a sign of {s}" for s in SUPERSTITIONS]
                        text = clip.tokenize(prompts).to(device)

                        with torch.no_grad():
                            image_features = model.encode_image(image)
                            text_features = model.encode_text(text)
                            logits = image_features @ text_features.T
                            probs = logits.softmax(dim=-1).cpu().numpy().flatten()

                        top_idx = int(probs.argmax())
                        stereotype = SUPERSTITIONS[top_idx]
                        counter = OPPOSITES[stereotype]

                        data.append({
                            "image_path": image_path,
                            "neutral_prompt": f"Image of a {base_term}",
                            "stereotype_prompt": prompts[top_idx],
                            "counter_prompt": f"Image of a {base_term} which is a sign of {counter}"
                        })
                    except Exception as e:
                        print(f"❌ Failed on {image_path}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset saved to {output_csv} with {len(df)} samples")
