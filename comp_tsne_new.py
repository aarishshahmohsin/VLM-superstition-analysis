import os
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vanilla CLIP model
vanilla_model, vanilla_preprocess = clip.load("ViT-B/32", device=device)
vanilla_model.eval()

# Load checkpoint model
checkpoint_model, checkpoint_preprocess = clip.load("ViT-B/32", device=device)
checkpoint_model.load_state_dict(
    torch.load("/home/aarish/VLM-superstition-analysis/models/clip_fold1_best.pt")["model"]
)
checkpoint_model.eval()

# === Helper functions ===
def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)

def get_image_embedding(model, preprocess, image_path: str) -> torch.Tensor | None:
    try:
        image = Image.open(image_path)
        if image.mode == "L":
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Grayish filter
        sampled_pixels = list(image.getdata())[::500]
        threshold = 10
        grayish = all(
            abs(r - g) < threshold and abs(g - b) < threshold and abs(r - b) < threshold
            for r, g, b in sampled_pixels
        )
        if grayish:
            return None

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image_tensor)
            return normalize(emb)
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def load_image_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images")
    return df

# === State → Region mapping ===
state_to_region = {
    # --- North India ---
    "delhi": "North",
    "haryana": "North",
    "punjab": "North",
    "himachal_pradesh": "North",
    "uttarakhand": "North",
    "jammu_and_kashmir": "North",
    "ladakh": "North",

    # --- Central India ---
    "madhya_pradesh": "Central",
    "chhattisgarh": "Central",

    # --- West India ---
    "rajasthan": "West",
    "gujarat": "West",
    "maharashtra": "West",
    "goa": "West",

    # --- South India ---
    "karnataka": "South",
    "kerala": "South",
    "tamil_nadu": "South",
    "andhra_pradesh": "South",
    "telangana": "South",
    "puducherry": "South",

    # --- East India ---
    "bihar": "East",
    "jharkhand": "East",
    "odisha": "East",
    "west_bengal": "East",

    # --- Northeast India ---
    "assam": "Northeast",
    "arunachal_pradesh": "Northeast",
    "manipur": "Northeast",
    "meghalaya": "Northeast",
    "mizoram": "Northeast",
    "nagaland": "Northeast",
    "tripura": "Northeast",
    "sikkim": "Northeast",

    # --- Islands & UTs ---
    "andaman_and_nicobar_islands": "Islands",
    "lakshadweep": "Islands",
    "chandigarh": "UT",
    "dadra_and_nagar_haveli_and_daman_and_diu": "UT",
}

# === Main function ===
def generate_tsne(csv_path: str):
    df = load_image_metadata(csv_path)

    checkpoint_embeddings = []
    vanilla_embeddings = []
    states = []

    print("Extracting embeddings...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row["image_path"]
        state = row["state"]

        checkpoint_emb = get_image_embedding(checkpoint_model, checkpoint_preprocess, image_path)
        vanilla_emb = get_image_embedding(vanilla_model, vanilla_preprocess, image_path)

        if checkpoint_emb is not None and vanilla_emb is not None:
            checkpoint_embeddings.append(checkpoint_emb.cpu().numpy())
            vanilla_embeddings.append(vanilla_emb.cpu().numpy())
            states.append(state)

    checkpoint_embeddings = np.vstack(checkpoint_embeddings)
    vanilla_embeddings = np.vstack(vanilla_embeddings)
    states = np.array(states)

    print(f"Successfully processed {len(states)} images")

    # --- Difference embeddings ---
    diff_embeddings = vanilla_embeddings - checkpoint_embeddings

    # --- Convert states → regions ---
    regions = np.array([state_to_region[s] for s in states])

    # --- TSNE ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    diff_tsne = tsne.fit_transform(diff_embeddings)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=diff_tsne[:, 0],
        y=diff_tsne[:, 1],
        hue=regions,
        palette="Set2",
        s=70,
        alpha=0.8,
    )
    plt.title("TSNE of Embedding Differences (Vanilla - Checkpoint) by Region")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Region")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# === Run ===
if __name__ == "__main__":
    csv_input = "/home/aarish/VLM-superstition-analysis/dataset_total.csv"
    generate_tsne(csv_input)
