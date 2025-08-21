import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# === Load embeddings ===
embeddings_1 = np.load("./similarities_array1.npy").squeeze(1)  # (2923, 512)
embeddings_2 = np.load("./similarities_array2.npy").squeeze(1)  # (2923, 512)

embeddings = embeddings_1 - embeddings_2

# === PCA before TSNE ===
pca = PCA(n_components=min(50, embeddings.shape[1]-1), svd_solver="randomized", random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

# === t-SNE ===
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    max_iter=1000,
    init="pca",
    method="exact",
    random_state=42,
    verbose=1
)
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# === Load metadata (states) ===
df = pd.read_csv("/home/aarish/VLM-superstition-analysis/dataset_total.csv")  # path to your CSV with state column
states = np.load('./states.npy')
# === Map each state to a unique color ===
unique_states = sorted(df["state"].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_states)))  # tab20 has 20 colors, cycles if >20
state_to_color = {state: colors[i % len(colors)] for i, state in enumerate(unique_states)}

# === Plot ===
plt.figure(figsize=(10, 8))
for state in unique_states:
    idx = df["state"] == state
    plt.scatter(
        embeddings_tsne[idx, 0],
        embeddings_tsne[idx, 1],
        s=10, alpha=0.7,
        color=state_to_color[state],
        label=state
    )

plt.title("t-SNE Visualization Colored by State")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
