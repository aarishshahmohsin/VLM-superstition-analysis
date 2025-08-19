import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px

# =========================
# 1. Load CSV
# =========================
csv_path = "/home/aarish/VLM-superstition-analysis/all_image_prompt_scores.csv"  # <-- update path
df = pd.read_csv(csv_path)

# Check first rows
print(df.head())

# =========================
# 2. Dataset stats
# =========================
print("Samples per state:\n", df['state'].value_counts())
print("Samples per gender:\n", df['gender'].value_counts())
print("Cross tab (gender per state):\n", pd.crosstab(df['state'], df['gender']))

# =========================
# 3. Extract embeddings
# =========================
# Assuming last 3 columns are ['image_path', 'gender', 'state']
embedding_cols = df.columns[:-3]  # adjust if different
embeddings = df[embedding_cols].values

# =========================
# 4. t-SNE 2D reduction
# =========================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)
df['x'] = embeddings_2d[:,0]
df['y'] = embeddings_2d[:,1]

# =========================
# 5. Plot t-SNE colored by gender
# =========================
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='x', y='y', hue='gender', palette='Set1', alpha=0.7)
plt.title("t-SNE visualization of embeddings colored by gender")
plt.show()

# =========================
# 6. Plot t-SNE colored by state
# =========================
plt.figure(figsize=(12,10))
sns.scatterplot(data=df, x='x', y='y', hue='state', palette='tab20', legend=False, alpha=0.7)
plt.title("t-SNE visualization of embeddings colored by state")
plt.show()

# =========================
# 7. Plot t-SNE colored by state+gender
# =========================
df['state_gender'] = df['state'] + "_" + df['gender']
plt.figure(figsize=(12,10))
sns.scatterplot(data=df, x='x', y='y', hue='state_gender', palette='tab20', legend=False, alpha=0.7)
plt.title("t-SNE: embeddings colored by state and gender")
plt.show()

# =========================
# 8. Display sample images
# =========================
sample = df.sample(9)  # random 9 images
fig, axes = plt.subplots(3, 3, figsize=(9,9))
for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
    img = Image.open(row['image_path'])
    ax.imshow(img)
    ax.set_title(f"{row['state']} - {row['gender']}", fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.show()

# =========================
# 9. Optional: Interactive Plotly scatter with hover images
# =========================
df['image_uri'] = df['image_path']  # plotly needs URI strings
fig = px.scatter(df, x='x', y='y', color='state', hover_data=['gender', 'state', 'image_uri'],
                 title="Interactive t-SNE scatter (hover to see image)", width=1000, height=800)
fig.show()
