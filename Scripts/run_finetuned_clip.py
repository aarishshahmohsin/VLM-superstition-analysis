import os
import torch
import pandas as pd
import clip
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

# Optional: Enable Colab download
try:
    from google.colab import files
    colab = True
except ImportError:
    colab = False

# ---------------------- Dataset ----------------------
class SuperstitionBiasDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        self.df = dataframe.reset_index(drop=True)
        self.transform = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        try:
            image = self.transform(Image.open(image_path).convert("RGB"))
        except Exception as e:
            print(f"❌ Skipping unreadable image: {image_path} — {e}")
            return self.__getitem__((idx + 1) % len(self.df))

        return {
            "image": image,
            "true_caption": row["neutral_prompt"],
            "stereotype_caption": row["stereotype_prompt"],
            "counter_caption": row["counter_prompt"]
        }

# ---------------------- Model Setup ----------------------
def load_model(model_name, device):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    return model.float().to(device), preprocess

def safe_tokenize(batch_texts):
    try:
        return clip.tokenize(batch_texts, truncate=True)
    except Exception as e:
        print("⚠️ Tokenization failed:", batch_texts, e)
        return clip.tokenize(["image"], truncate=True)

# ---------------------- Contrastive Loss ----------------------
def contrastive_loss(image_features, pos, neg1, neg2, temperature=0.1, eps=1e-8):
    image_features = F.normalize(image_features, dim=-1, eps=eps)
    pos = F.normalize(pos, dim=-1, eps=eps)
    neg1 = F.normalize(neg1, dim=-1, eps=eps)
    neg2 = F.normalize(neg2, dim=-1, eps=eps)

    sim_pos = (image_features * pos).sum(dim=1)
    sim_neg1 = (image_features * neg1).sum(dim=1)
    sim_neg2 = (image_features * neg2).sum(dim=1)

    logits = torch.stack([sim_pos, sim_neg1, sim_neg2], dim=1) / temperature
    labels = torch.zeros(image_features.size(0), dtype=torch.long).to(image_features.device)
    return nn.CrossEntropyLoss()(logits, labels)

# ---------------------- Training ----------------------
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        true = safe_tokenize(batch["true_caption"]).to(device)
        stereo = safe_tokenize(batch["stereotype_caption"]).to(device)
        counter = safe_tokenize(batch["counter_caption"]).to(device)

        image_features = model.encode_image(images).float()
        tf_pos = model.encode_text(true).float()
        tf_neg1 = model.encode_text(stereo).float()
        tf_neg2 = model.encode_text(counter).float()

        loss = contrastive_loss(image_features, tf_pos, tf_neg1, tf_neg2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# ---------------------- Evaluation ----------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total, correct = 0, 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        true = model.encode_text(safe_tokenize(batch["true_caption"]).to(device)).float()
        stereo = model.encode_text(safe_tokenize(batch["stereotype_caption"]).to(device)).float()
        counter = model.encode_text(safe_tokenize(batch["counter_caption"]).to(device)).float()
        image_features = model.encode_image(images).float()

        image_features = F.normalize(image_features, dim=-1)
        true = F.normalize(true, dim=-1)
        stereo = F.normalize(stereo, dim=-1)
        counter = F.normalize(counter, dim=-1)

        sim_true = (image_features * true).sum(dim=1)
        sim_stereo = (image_features * stereo).sum(dim=1)
        sim_counter = (image_features * counter).sum(dim=1)

        correct += (sim_true > sim_stereo).sum().item()
        correct += (sim_true > sim_counter).sum().item()
        total += 2 * images.size(0)

    return round(correct / total * 100, 2)

# ---------------------- K-Fold CV ----------------------
def run_kfold_training(args):
    df = pd.read_csv(args.csv_path).dropna()
    df = df[(df["neutral_prompt"].str.strip() != "") &
            (df["stereotype_prompt"].str.strip() != "") &
            (df["counter_prompt"].str.strip() != "")].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n🌀 Fold {fold + 1}/{args.k_folds}")
        model, preprocess = load_model(args.model_name, device)

        train_ds = SuperstitionBiasDataset(df.iloc[train_idx], preprocess)
        val_ds = SuperstitionBiasDataset(df.iloc[val_idx], preprocess)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=64, num_workers=2)

        for param in model.visual.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=args.lr, weight_decay=0.2)

        for epoch in range(args.epochs):
            loss = train_model(model, train_dl, optimizer, device)
            print(f"📉 Epoch {epoch+1} - Loss: {loss:.4f}")

        acc = evaluate(model, val_dl, device)
        fold_accuracies.append(acc)
        print(f"✅ Fold {fold+1} Accuracy: {acc}%")

        fold_model_path = os.path.join(args.save_path, f"clip_fold{fold+1}.pt")
        torch.save({"model": model.state_dict()}, fold_model_path)

        # Save last fold as final model
        if fold + 1 == args.k_folds:
            final_model_path = os.path.join(args.save_path, "superstition_clip_final.pt")
            torch.save({"model": model.state_dict()}, final_model_path)
            print(f"📦 Final model saved at: {final_model_path}")
            if colab:
                files.download(final_model_path)

    print(f"\n🎯 Average Accuracy: {sum(fold_accuracies)/args.k_folds:.2f}%")
    return fold_accuracies

# ---------------------- Args ----------------------
if __name__ == "__main__":
    class Args:
        csv_path = "/content/clip_superstition_dataset.csv"
        model_name = "ViT-B/32"
        batch_size = 16
        epochs = 3
        lr = 1e-5
        k_folds = 5
        save_path = "/content/models"  # works in Colab or locally

    os.makedirs(Args.save_path, exist_ok=True)
    run_kfold_training(Args)
