import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

embeddings_1 = np.load("./similarities_array1.npy")
embeddings_2 = np.load("./similarities_array2.npy")



embeddings_1 = embeddings_1.squeeze(1)  # -> (2923, 512)
embeddings_2 = embeddings_2.squeeze(1)  # -> (2923, 512)

embeddings = embeddings_1 - embeddings_2
for i in embeddings:
    print(i)
# embeddings = StandardScaler().fit_transform(embeddings)

# pca = PCA(n_components=min(50, embeddings.shape[1]-1), svd_solver="randomized", random_state=42)
# embeddings_pca = pca.fit_transform(embeddings)


# # Step 2: TSNE
# tsne = TSNE(
#     n_components=2,
#     perplexity=30,
#     learning_rate="auto",
#     max_iter=1000,
#     init="pca",
#     method="exact",     # ✅ more stable for ~3k samples
#     random_state=42,
#     verbose=1
# )

# embeddings_tsne = tsne.fit_transform(embeddings_pca)

# # Step 3: Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=10, alpha=0.7)
# plt.title("t-SNE Visualization")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.show()