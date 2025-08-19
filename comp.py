import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vanilla CLIP model
vanilla_model, vanilla_preprocess = clip.load("ViT-B/32", device=device)
vanilla_model.eval()

# Load checkpoint model (can just be a separately loaded CLIP for demonstration)
# If you have a fine-tuned checkpoint, load weights here
checkpoint_model, checkpoint_preprocess = clip.load("ViT-B/32", device=device)
checkpoint_model.load_state_dict(torch.load('/home/aarish/VLM-superstition-analysis/models/superstition_clip_final.pt')['model'])
checkpoint_model.eval()

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)

def get_text_embedding(model, text: str) -> torch.Tensor:
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        return normalize(embedding)

def get_image_embedding(model, preprocess, image_path: str) -> torch.Tensor | None:
    try:
        image = Image.open(image_path)

        if image.mode == "L":
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")

        sampled_pixels = list(image.getdata())[::500]
        threshold = 10
        grayish = all(abs(r - g) < threshold and abs(g - b) < threshold and abs(r - b) < threshold
                      for r, g, b in sampled_pixels)
        if grayish:
            return None

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            return normalize(embedding)

    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def load_image_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    def get_image_number(path):
        try:
            filename = os.path.basename(path)
            return int(filename.split('.')[0])
        except:
            return 0
    df['image_number'] = df['image_path'].apply(get_image_number)
    df_filtered = df  # no filtering
    print(f"Loaded {len(df_filtered)} images (filtered from {len(df)} total)")
    return df_filtered

def compare_and_plot_simplified(csv_path: str, output_csv: str):
    df = load_image_metadata(csv_path)
    
    india_emb_checkpoint = get_text_embedding(checkpoint_model, "this is a face of an Indian")
    india_emb_vanilla = get_text_embedding(vanilla_model, "this is a face of an Indian")
    
    checkpoint_embeddings = []
    vanilla_embeddings = []
    states = []
    checkpoint_similarities = []
    vanilla_similarities = []
    difference = []

    print("Processing images with both models...")
    for idx, row in tqdm(df.iterrows(), desc="Processing images", total=len(df)):
        image_path = row['image_path']
        state = row['state']
        
        checkpoint_emb = get_image_embedding(checkpoint_model, checkpoint_preprocess, image_path)
        vanilla_emb = get_image_embedding(vanilla_model, vanilla_preprocess, image_path)
        
        if checkpoint_emb is not None and vanilla_emb is not None:
            checkpoint_sim = (checkpoint_emb @ india_emb_checkpoint.T).item()
            vanilla_sim = (vanilla_emb @ india_emb_vanilla.T).item()
            
            checkpoint_embeddings.append(checkpoint_emb.cpu().numpy())
            vanilla_embeddings.append(vanilla_emb.cpu().numpy())
            states.append(state)
            checkpoint_similarities.append(checkpoint_sim)
            vanilla_similarities.append(vanilla_sim)
            difference.append(vanilla_sim-checkpoint_sim)

    # Convert to numpy arrays
    checkpoint_embeddings = np.vstack(checkpoint_embeddings)
    vanilla_embeddings = np.vstack(vanilla_embeddings)
    states = np.array(states)
    checkpoint_similarities = np.array(checkpoint_similarities)
    vanilla_similarities = np.array(vanilla_similarities)
    difference = np.array(difference)

    np.save("similarities_array", difference)

    print(f"Successfully processed {len(states)} images")
    print(f"States found: {np.unique(states)}")

    unique_states = np.unique(states)
    
    checkpoint_avg_sim_per_state = {}
    checkpoint_sim_of_avg_emb_per_state = {}
    vanilla_avg_sim_per_state = {}
    vanilla_sim_of_avg_emb_per_state = {}

    for s in unique_states:
        mask = states == s
        
        checkpoint_avg_emb = checkpoint_embeddings[mask].mean(axis=0)
        checkpoint_avg_sim_per_state[s] = checkpoint_similarities[mask].mean()
        checkpoint_sim_of_avg_emb_per_state[s] = (normalize(torch.tensor(checkpoint_avg_emb, device=device)) @ india_emb_checkpoint.T).item()
        
        vanilla_avg_emb = vanilla_embeddings[mask].mean(axis=0)
        vanilla_avg_sim_per_state[s] = vanilla_similarities[mask].mean()
        vanilla_sim_of_avg_emb_per_state[s] = (normalize(torch.tensor(vanilla_avg_emb, device=device)) @ india_emb_vanilla.T).item()

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["state", "checkpoint_avg_similarity", "checkpoint_similarity_of_avg_embedding", 
                        "vanilla_avg_similarity", "vanilla_similarity_of_avg_embedding"])
        for s in unique_states:
            writer.writerow([s, f"{checkpoint_avg_sim_per_state[s]:.6f}", 
                             f"{checkpoint_sim_of_avg_emb_per_state[s]:.6f}",
                             f"{vanilla_avg_sim_per_state[s]:.6f}", 
                             f"{vanilla_sim_of_avg_emb_per_state[s]:.6f}"])
    print(f"✅ Saved state-wise summary to {output_csv}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x_pos = np.arange(len(unique_states))
    width = 0.35
    
    checkpoint_avg_sims = [checkpoint_avg_sim_per_state[s] for s in unique_states]
    vanilla_avg_sims = [vanilla_avg_sim_per_state[s] for s in unique_states]
    
    ax1.bar(x_pos - width/2, checkpoint_avg_sims, width, label='Checkpoint Model', alpha=0.7, color='blue')
    ax1.bar(x_pos + width/2, vanilla_avg_sims, width, label='Vanilla Model', alpha=0.7, color='orange')
    ax1.set_xlabel('States')
    ax1.set_ylabel('Average Similarity')
    ax1.set_title('Average of Similarities to "an indian person"')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(unique_states, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    checkpoint_sim_avgs = [checkpoint_sim_of_avg_emb_per_state[s] for s in unique_states]
    vanilla_sim_avgs = [vanilla_sim_of_avg_emb_per_state[s] for s in unique_states]
    
    ax2.bar(x_pos - width/2, checkpoint_sim_avgs, width, label='Checkpoint Model', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, vanilla_sim_avgs, width, label='Vanilla Model', alpha=0.7, color='orange')
    ax2.set_xlabel('States')
    ax2.set_ylabel('Similarity of Average Embedding')
    ax2.set_title('Similarity of Average Embeddings to "an indian person"')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(unique_states, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return {
        'checkpoint_avg_sim_per_state': checkpoint_avg_sim_per_state,
        'vanilla_avg_sim_per_state': vanilla_avg_sim_per_state,
        'checkpoint_sim_of_avg_emb_per_state': checkpoint_sim_of_avg_emb_per_state,
        'vanilla_sim_of_avg_emb_per_state': vanilla_sim_of_avg_emb_per_state,
        'unique_states': unique_states
    }

if __name__ == "__main__":
    csv_input = "/home/aarish/VLM-superstition-analysis/dataset_total.csv"
    output_csv = "dual_model_statewise_similarity.csv"
    results = compare_and_plot_simplified(csv_input, output_csv)
