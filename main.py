# main.py

import os

if __name__ == "__main__":
    print("🚀 Starting Superstition Bias CLIP Pipeline")

    # STEP 1: Download and extract dataset (if using Kaggle or external source)
    # For example, if using KaggleHub (adjust if dataset is already available):
    # from data_download import download_dataset
    # download_dataset()

    # STEP 2: Run Zero-Shot Evaluation with ViT-B/32
    os.system("python run_vit_b32_zeroshot.py")

    # STEP 3: Run Zero-Shot Evaluation with ViT-L/14
    os.system("python run_vit_l14_zeroshot.py")

    # STEP 4: Create dataset for fine-tuning
    os.system("python create_finetune_dataset.py")

    # STEP 5: Fine-tune CLIP using the created dataset
    os.system("python finetune_clip.py")

    # STEP 6: Evaluate Fine-Tuned model
    os.system("python run_finetuned_clip.py")

    # STEP 7: Visualization and value count comparison
    os.system("python visualize_results.py")

    print("\n✅ All stages completed successfully.")
