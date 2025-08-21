import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# Define texts
# texts = ["person from India", "person from foreign"]
texts = ["person from India", "person from foreign"]

# Tokenize
text_tokens = clip.tokenize(texts).to(device)

# Encode
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)

# Normalize
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# Convert to numpy for sklearn
emb_np = text_embeddings.cpu().numpy()

# Cosine similarity
similarity = cosine_similarity(emb_np)
print("Cosine Similarity Matrix:")
print(similarity)

# Direct similarity between Indian & Non-Indian
print(f"\nSimilarity(Indian, Non-Indian): {similarity[0,1]:.4f}")

