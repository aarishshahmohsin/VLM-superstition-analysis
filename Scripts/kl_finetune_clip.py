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
import random
import json

# ---------------------- Datasets ----------------------
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

class NoisyImageDataset(Dataset):
    """Imagenet-style dataset: just returns images for KL consistency."""
    def __init__(self, dataframe, preprocess, base_path):
        self.df = dataframe.reset_index(drop=True)
        self.transform = preprocess
        self.base_path = base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.base_path, row["path"])
        try:
            image = self.transform(Image.open(image_path).convert("RGB"))
        except Exception as e:
            print(f"❌ Skipping unreadable noisy image: {image_path} — {e}")
            return self.__getitem__((idx + 1) % len(self.df))

        return {"image": image}

# ---------------------- Model Setup ----------------------
def load_model(model_name, device, trainable=True):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model = model.float()
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model.to(device), preprocess

def safe_tokenize(batch_texts):
    try:
        return clip.tokenize(batch_texts, truncate=True)
    except Exception as e:
        print("⚠️ Tokenization failed:", batch_texts, e)
        return clip.tokenize(["image"], truncate=True)

# ---------------------- Text Sampling ----------------------
def sample_random_texts(batch_size):
    """Generate random gibberish strings."""
    random_texts = ["random_" + str(random.randint(0, 99999)) for _ in range(batch_size)]
    return clip.tokenize(random_texts, truncate=True)

# A small subset of ImageNet classes (you can replace with full 1k list)
IMAGENET_CLASSES = [
    "dog", "cat", "car", "tree", "building", "person", "airplane",
    "flower", "bicycle", "horse", "boat", "bird", "fish", "fruit", "chair"
]

def sample_imagenet_labels(batch_size):
    """Sample ImageNet class names as text prompts."""
    sampled = random.choices(IMAGENET_CLASSES, k=batch_size)
    return clip.tokenize(sampled, truncate=True)

# ---------------------- Combined Loss Function ----------------------
def combined_contrastive_kl_loss(
    student_model,
    teacher_model,
    image_features,
    pos,
    neg1,
    neg2,
    noisy_images,
    noisy_texts=None,
    temperature=0.3,
    eps=1e-8,
    kl_weight_img=0.5,
    kl_weight_txt=0.5
):
    """
    Combined loss:
    1. Contrastive CE loss for superstition bias (main task)
    2. KL divergence regularization (images)
    3. KL divergence regularization (texts)
    """

    # ===== Main Contrastive Loss =====
    image_features = F.normalize(image_features, dim=-1, eps=eps)
    pos = F.normalize(pos, dim=-1, eps=eps)
    neg1 = F.normalize(neg1, dim=-1, eps=eps)
    neg2 = F.normalize(neg2, dim=-1, eps=eps)

    sim_pos = torch.einsum('bd,bd->b', image_features, pos)
    sim_neg1 = torch.einsum('bd,bd->b', image_features, neg1)
    sim_neg2 = torch.einsum('bd,bd->b', image_features, neg2)

    logits = torch.stack([sim_pos, sim_neg1, sim_neg2], dim=1) / temperature
    logits = torch.clamp(logits, min=-10, max=10)

    labels = torch.zeros(image_features.size(0), dtype=torch.long, device=image_features.device)
    ce_loss = F.cross_entropy(logits, labels)

    # ===== Image KL Loss =====
    kl_img_loss = 0.0
    if noisy_images is not None:
        with torch.no_grad():
            teacher_features = F.normalize(teacher_model.encode_image(noisy_images).float(), dim=-1, eps=eps)
        student_features = F.normalize(student_model.encode_image(noisy_images).float(), dim=-1, eps=eps)

        student_logits = torch.matmul(student_features, student_features.T) / temperature
        teacher_logits = torch.matmul(teacher_features, teacher_features.T) / temperature

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        kl_img_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # ===== Text KL Loss =====
    kl_txt_loss = 0.0
    if noisy_texts is not None:
        with torch.no_grad():
            teacher_text = F.normalize(teacher_model.encode_text(noisy_texts).float(), dim=-1, eps=eps)
        student_text = F.normalize(student_model.encode_text(noisy_texts).float(), dim=-1, eps=eps)

        student_logits_t = torch.matmul(student_text, student_text.T) / temperature
        teacher_logits_t = torch.matmul(teacher_text, teacher_text.T) / temperature

        student_log_probs_t = F.log_softmax(student_logits_t, dim=-1)
        teacher_probs_t = F.softmax(teacher_logits_t, dim=-1)

        kl_txt_loss = F.kl_div(student_log_probs_t, teacher_probs_t, reduction="batchmean")

    # ===== Total Loss =====
    total_loss = ce_loss + kl_weight_img * kl_img_loss + kl_weight_txt * kl_txt_loss
    return total_loss, ce_loss, kl_img_loss, kl_txt_loss

# ---------------------- Training ----------------------
def train_model_with_noisy(student_model, teacher_model, main_dataloader, noisy_dataloader, optimizer, device, max_grad_norm=1.0):
    student_model.train()
    total_loss, total_ce_loss, total_kl_img_loss, total_kl_txt_loss, num_batches = 0, 0, 0, 0, 0

    noisy_iter = iter(noisy_dataloader)

    for batch in tqdm(main_dataloader, desc="Training"):
        try:
            images = batch["image"].to(device)
            true = safe_tokenize(batch["true_caption"]).to(device)
            stereo = safe_tokenize(batch["stereotype_caption"]).to(device)
            counter = safe_tokenize(batch["counter_caption"]).to(device)

            try:
                noisy_batch = next(noisy_iter)
            except StopIteration:
                noisy_iter = iter(noisy_dataloader)
                noisy_batch = next(noisy_iter)

            noisy_images = noisy_batch["image"].to(device)

            # Option 1: Random gibberish
            # noisy_texts = sample_random_texts(noisy_images.size(0)).to(device)

            # Option 2: ImageNet class names
            noisy_texts = sample_imagenet_labels(noisy_images.size(0)).to(device)

            with torch.cuda.amp.autocast(enabled=False):
                image_features = student_model.encode_image(images).float()
                tf_pos = student_model.encode_text(true).float()
                tf_neg1 = student_model.encode_text(stereo).float()
                tf_neg2 = student_model.encode_text(counter).float()

                loss, ce_loss, kl_img_loss, kl_txt_loss = combined_contrastive_kl_loss(
                    student_model,
                    teacher_model,
                    image_features,
                    tf_pos,
                    tf_neg1,
                    tf_neg2,
                    noisy_images,
                    noisy_texts=noisy_texts,
                    kl_weight_img=1000,   # increased
                    kl_weight_txt=1000    # new
                )

            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kl_img_loss += kl_img_loss.item() if isinstance(kl_img_loss, torch.Tensor) else kl_img_loss
            total_kl_txt_loss += kl_txt_loss.item() if isinstance(kl_txt_loss, torch.Tensor) else kl_txt_loss
            num_batches += 1

        except Exception as e:
            print(f"⚠️ Error in training batch: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    avg_ce_loss = total_ce_loss / max(num_batches, 1)
    avg_kl_img_loss = total_kl_img_loss / max(num_batches, 1)
    avg_kl_txt_loss = total_kl_txt_loss / max(num_batches, 1)

    return avg_loss, avg_ce_loss, avg_kl_img_loss, avg_kl_txt_loss

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

# ---------------------- K-Fold CV with Noisy Labels ----------------------
def run_kfold_training_with_noisy(args):
    df = pd.read_csv(args.csv_path).dropna()
    df = df[(df["neutral_prompt"].str.strip() != "") &
            (df["stereotype_prompt"].str.strip() != "") &
            (df["counter_prompt"].str.strip() != "")].reset_index(drop=True)
    print(f"📊 Main dataset size: {len(df)} samples")

    noisy_df = pd.read_csv(args.noisy_csv_path).dropna()
    print(f"📊 Noisy dataset size: {len(noisy_df)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n🌀 Fold {fold + 1}/{args.k_folds}")
        student_model, preprocess = load_model(args.model_name, device, trainable=True)
        teacher_model, _ = load_model(args.model_name, device, trainable=False)

        train_ds = SuperstitionBiasDataset(df.iloc[train_idx], preprocess)
        val_ds = SuperstitionBiasDataset(df.iloc[val_idx], preprocess)

        noisy_sample = noisy_df.sample(n=min(args.noisy_sample_size, len(noisy_df)), random_state=fold+42)
        noisy_ds = NoisyImageDataset(noisy_sample, preprocess, args.imagenette_path)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)
        noisy_dl = DataLoader(noisy_ds, batch_size=args.noisy_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        print(f"📊 Fold {fold+1}: Train={len(train_ds)}, Val={len(val_ds)}, Noisy={len(noisy_ds)}")

        # Using Adam optimizer with constant learning rate
        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=args.lr, 
            betas=(0.9, 0.98), 
            eps=1e-6
        )

        # No scheduler - keeping constant learning rate
        # Removed the cosine schedule with warmup

        best_acc = 0
        for epoch in range(args.epochs):
            loss, ce_loss, kl_img_loss, kl_txt_loss = train_model_with_noisy(student_model, teacher_model, train_dl, noisy_dl, optimizer, device)
            # No scheduler.step() call - LR remains constant
            acc = evaluate(student_model, val_dl, device)
            print(f"📉 Epoch {epoch+1} - Total: {loss:.4f}, CE: {ce_loss:.4f}, KL_img: {kl_img_loss:.4f}, KL_txt: {kl_txt_loss:.4f}, Acc: {acc}%, LR: {optimizer.param_groups[0]['lr']:.2e}")

            if acc > best_acc:
                best_acc = acc
                best_fold_path = os.path.join(args.save_path, f"clip_fold{fold+1}_best.pt")
                torch.save({"model": student_model.state_dict(), "accuracy": acc, "epoch": epoch+1}, best_fold_path)

        fold_accuracies.append(best_acc)
        print(f"✅ Fold {fold+1} Best Accuracy: {best_acc}%")

        if fold + 1 == args.k_folds:
            final_model_path = os.path.join(args.save_path, "superstition_clip_final_with_noisy.pt")
            torch.save({"model": student_model.state_dict(), "fold_accuracies": fold_accuracies, "final_accuracy": best_acc}, final_model_path)
            print(f"📦 Final model saved at: {final_model_path}")

    avg_accuracy = sum(fold_accuracies)/args.k_folds
    print(f"\n🎯 Average Accuracy: {avg_accuracy:.2f}%")
    print(f"📈 Fold Accuracies: {fold_accuracies}")
    return fold_accuracies

# ---------------------- Args ----------------------
if __name__ == "__main__":
    class Args:
        csv_path = "/home/aarish/VLM-superstition-analysis/indian_dataset.csv"
        noisy_csv_path = "/home/aarish/VLM-superstition-analysis/image_dataset.csv"
        imagenette_path = "/mnt/c/Users/HP/Downloads/archive"
        noisy_sample_size = 1000
        noisy_batch_size = 8
        model_name = "ViT-B/32"
        batch_size = 16
        epochs = 10
        lr = 5e-5  # This will remain constant throughout training
        k_folds = 5
        save_path = "./models"

    os.makedirs(Args.save_path, exist_ok=True)
    run_kfold_training_with_noisy(Args)