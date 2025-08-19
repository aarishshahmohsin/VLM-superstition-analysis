# run_finetuned_clip_csv_batch_safe.py

import os
import pandas as pd
import torch
import clip
from PIL import Image

# === 1. Setup ===
csv_path = "/home/aarish/VLM-superstition-analysis/indian_dataset.csv"  # Your CSV
model_path = '/home/aarish/VLM-superstition-analysis/models/superstition_clip_final.pt'
output_base_dir = "fine_tuned_results_india_csv_batch_safe"

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Adjust based on your GPU memory

# === 2. Load fine-tuned CLIP ===
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model"])
model = model.float().to(device).eval()
print("✅ Loaded fine-tuned CLIP model.")

# === 3. Load CSV dataset ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=["image_path", "gender", "state"])
print(f"✅ Loaded {len(df)} rows from CSV.")

# === 4. Preprocess images ===
preprocessed_images = []
valid_image_paths = []
valid_genders = []
valid_states = []

for idx, row in df.iterrows():
    path = row["image_path"]
    gender = row["gender"]
    state = row["state"]
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        preprocessed_images.append(image)
        valid_image_paths.append(path)
        valid_genders.append(gender)
        valid_states.append(state)
    except Exception as e:
        print(f"❌ Failed image {path}: {e}")

if not preprocessed_images:
    raise RuntimeError("No valid images found!")

# === 5. Encode images in batches to avoid OOM ===
image_features_list = []
for i in range(0, len(preprocessed_images), BATCH_SIZE):
    batch = torch.cat(preprocessed_images[i:i+BATCH_SIZE])
    with torch.no_grad():
        feats = model.encode_image(batch).float()
        feats /= feats.norm(dim=-1, keepdim=True)
        image_features_list.append(feats)
    # Free GPU memory for batch
    del batch, feats
    torch.cuda.empty_cache()

image_features = torch.cat(image_features_list)

# === 6. Generate prompts ===
unique_states = sorted(set(valid_states))
genders = ["male", "female", "other"]


prompts = [f"Image of a person from {state} who is {gender}" 
           for state in unique_states for gender in genders]
text_tokens = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

# === 7. Compute similarity ===
temperature = 0.1
logits = image_features @ text_features.T
logits /= temperature
probs = logits.softmax(dim=-1)  # shape: [num_images, num_prompts]

# === 8. Save top matches per prompt ===
os.makedirs(output_base_dir, exist_ok=True)

for prompt_idx, prompt in enumerate(prompts):
    prompt_safe = prompt.replace(" ", "_").replace("/", "_")
    prompt_dir = os.path.join(output_base_dir, f"{prompt_idx}_{prompt_safe}")
    os.makedirs(prompt_dir, exist_ok=True)

    top_k = 10
    top_indices = probs[:, prompt_idx].topk(top_k).indices
    top_scores = probs[:, prompt_idx].topk(top_k).values

    print(f"\n📌 Prompt: '{prompt}' — Top {top_k} matches:")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        img_path = valid_image_paths[idx.item()]
        print(f"{i+1}. {img_path} — Score: {score.item():.4f}")
        try:
            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(prompt_dir, f"{i+1}_score_{score.item():.4f}.jpg"))
        except Exception as e:
            print(f"Error saving: {img_path} — {e}")

# === 9. Save CSV of all scores ===
scores_df = pd.DataFrame(probs.cpu().numpy(), columns=prompts)
scores_df["image_path"] = valid_image_paths
scores_df["gender"] = valid_genders
scores_df["state"] = valid_states
scores_df.to_csv(os.path.join(output_base_dir, "all_image_prompt_scores.csv"), index=False)

print("\n✅ Batch-safe evaluation complete. Scores CSV saved.")
