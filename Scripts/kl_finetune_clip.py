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
    model = model.float()
    return model.to(device), preprocess

def safe_tokenize(batch_texts):
    try:
        return clip.tokenize(batch_texts, truncate=True)
    except Exception as e:
        print("⚠️ Tokenization failed:", batch_texts, e)
        return clip.tokenize(["image"], truncate=True)

# ---------------------- Contrastive Loss + KL Divergence ----------------------
def contrastive_loss_kl(image_features, pos, neg1, neg2, temperature=0.3, eps=1e-8, kl_weight=0.5):
    # Normalize
    image_features = F.normalize(image_features, dim=-1, eps=eps)
    pos = F.normalize(pos, dim=-1, eps=eps)
    neg1 = F.normalize(neg1, dim=-1, eps=eps)
    neg2 = F.normalize(neg2, dim=-1, eps=eps)

    # Similarities
    sim_pos = torch.einsum('bd,bd->b', image_features, pos)
    sim_neg1 = torch.einsum('bd,bd->b', image_features, neg1)
    sim_neg2 = torch.einsum('bd,bd->b', image_features, neg2)

    # Stack logits
    logits = torch.stack([sim_pos, sim_neg1, sim_neg2], dim=1) / temperature
    logits = torch.clamp(logits, min=-10, max=10)

    # Cross-entropy target
    labels = torch.zeros(image_features.size(0), dtype=torch.long, device=image_features.device)
    ce_loss = F.cross_entropy(logits, labels)

    # KL divergence: soft targets
    probs_pred = F.log_softmax(logits, dim=1)  # log-probs for KL
    # Soft target distribution: give more weight to positive
    target_probs = torch.tensor([[0.7, 0.15, 0.15]] * image_features.size(0), 
                                device=image_features.device)
    kl_loss = F.kl_div(probs_pred, target_probs, reduction='batchmean')

    return ce_loss + kl_weight * kl_loss

# ---------------------- Training ----------------------
def train_model(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        try:
            images = batch["image"].to(device)
            true = safe_tokenize(batch["true_caption"]).to(device)
            stereo = safe_tokenize(batch["stereotype_caption"]).to(device)
            counter = safe_tokenize(batch["counter_caption"]).to(device)

            with torch.cuda.amp.autocast(enabled=False):
                image_features = model.encode_image(images).float()
                tf_pos = model.encode_text(true).float()
                tf_neg1 = model.encode_text(stereo).float()
                tf_neg2 = model.encode_text(counter).float()

                loss = contrastive_loss_kl(image_features, tf_pos, tf_neg1, tf_neg2)

            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"⚠️ Error in training batch: {e}")
            continue

    return total_loss / max(num_batches, 1)

# ---------------------- Evaluation ----------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total, correct_vs_stereo, correct_vs_counter = 0, 0, 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        try:
            images = batch["image"].to(device)
            true = safe_tokenize(batch["true_caption"]).to(device)
            stereo = safe_tokenize(batch["stereotype_caption"]).to(device)
            counter = safe_tokenize(batch["counter_caption"]).to(device)

            image_features = F.normalize(model.encode_image(images).float(), dim=-1)
            tf_true = F.normalize(model.encode_text(true).float(), dim=-1)
            tf_stereo = F.normalize(model.encode_text(stereo).float(), dim=-1)
            tf_counter = F.normalize(model.encode_text(counter).float(), dim=-1)

            sim_true = torch.einsum('bd,bd->b', image_features, tf_true)
            sim_stereo = torch.einsum('bd,bd->b', image_features, tf_stereo)
            sim_counter = torch.einsum('bd,bd->b', image_features, tf_counter)

            correct_vs_stereo += (sim_true > sim_stereo).sum().item()
            correct_vs_counter += (sim_true > sim_counter).sum().item()
            total += images.size(0)

        except Exception as e:
            print(f"⚠️ Error in evaluation batch: {e}")
            continue

    accuracy = round((correct_vs_stereo + correct_vs_counter) / (2 * max(total, 1)) * 100, 2)
    return accuracy

# ---------------------- Learning Rate Scheduler ----------------------
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------------- K-Fold CV ----------------------
def run_kfold_training(args):
    df = pd.read_csv(args.csv_path).dropna()
    df = df[(df["neutral_prompt"].str.strip() != "") &
            (df["stereotype_prompt"].str.strip() != "") &
            (df["counter_prompt"].str.strip() != "")].reset_index(drop=True)
    print(f"📊 Dataset size: {len(df)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n🌀 Fold {fold + 1}/{args.k_folds}")
        model, preprocess = load_model(args.model_name, device)

        train_ds = SuperstitionBiasDataset(df.iloc[train_idx], preprocess)
        val_ds = SuperstitionBiasDataset(df.iloc[val_idx], preprocess)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)


        optimizer = torch.optim.AdamW(
            # filter(lambda p: p.requires_grad, model.parameters()), 
            model.parameters(),
            lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6
        )

        num_training_steps = len(train_dl) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(train_dl)//2, num_training_steps=num_training_steps
        )

        best_acc = 0
        for epoch in range(args.epochs):
            loss = train_model(model, train_dl, optimizer, device)
            scheduler.step()
            acc = evaluate(model, val_dl, device)
            print(f"📉 Epoch {epoch+1} - Loss: {loss:.4f}, Accuracy: {acc}%, LR: {optimizer.param_groups[0]['lr']:.2e}")

            if acc > best_acc:
                best_acc = acc
                best_fold_path = os.path.join(args.save_path, f"clip_fold{fold+1}_best.pt")
                torch.save({"model": model.state_dict(), "accuracy": acc, "epoch": epoch+1}, best_fold_path)

        fold_accuracies.append(best_acc)
        print(f"✅ Fold {fold+1} Best Accuracy: {best_acc}%")

        if fold + 1 == args.k_folds:
            final_model_path = os.path.join(args.save_path, "superstition_clip_final.pt")
            torch.save({"model": model.state_dict(), "fold_accuracies": fold_accuracies, "final_accuracy": best_acc}, final_model_path)
            print(f"📦 Final model saved at: {final_model_path}")
            if colab:
                files.download(final_model_path)

    avg_accuracy = sum(fold_accuracies)/args.k_folds
    print(f"\n🎯 Average Accuracy: {avg_accuracy:.2f}%")
    print(f"📈 Fold Accuracies: {fold_accuracies}")
    return fold_accuracies

# ---------------------- Debug Function ----------------------
def debug_similarities(model, dataloader, device, num_samples=5):
    model.eval()
    print("\n🔍 Debugging similarity values...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            images = batch["image"][:1].to(device)
            true = safe_tokenize([batch["true_caption"][0]]).to(device)
            stereo = safe_tokenize([batch["stereotype_caption"][0]]).to(device)
            counter = safe_tokenize([batch["counter_caption"][0]]).to(device)

            image_features = F.normalize(model.encode_image(images).float(), dim=-1)
            tf_true = F.normalize(model.encode_text(true).float(), dim=-1)
            tf_stereo = F.normalize(model.encode_text(stereo).float(), dim=-1)
            tf_counter = F.normalize(model.encode_text(counter).float(), dim=-1)

            sim_true = torch.einsum('bd,bd->b', image_features, tf_true)
            sim_stereo = torch.einsum('bd,bd->b', image_features, tf_stereo)
            sim_counter = torch.einsum('bd,bd->b', image_features, tf_counter)

            print(f"Sample {i+1}: True: {sim_true.item():.4f}, Stereo: {sim_stereo.item():.4f}, Counter: {sim_counter.item():.4f}")

# ---------------------- Args ----------------------
if __name__ == "__main__":
    class Args:
        csv_path = "/home/aarish/VLM-superstition-analysis/indian_dataset.csv"
        model_name = "ViT-B/32"
        batch_size = 16
        epochs = 10
        # lr = 5e-6
        lr = 1e-4
        k_folds = 5
        save_path = "./models"

    os.makedirs(Args.save_path, exist_ok=True)

    print("🧪 Running similarity debug...")
    df_debug = pd.read_csv(Args.csv_path).dropna().head(10)
    model_debug, preprocess_debug = load_model(Args.model_name, "cuda" if torch.cuda.is_available() else "cpu")
    debug_ds = SuperstitionBiasDataset(df_debug, preprocess_debug)
    debug_dl = DataLoader(debug_ds, batch_size=1, num_workers=0)
    debug_similarities(model_debug, debug_dl, "cuda" if torch.cuda.is_available() else "cpu")

    run_kfold_training(Args)
