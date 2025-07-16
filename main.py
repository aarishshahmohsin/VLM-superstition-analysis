
# Superstition Dataset CLIP Analysis and Fine-tuning Pipeline (Hugging Face model + Kaggle dataset)

import os
import argparse
from scripts import dataset_processing, clip_inference, fine_tuning, visualize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Superstition CLIP Model Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['infer', 'finetune', 'evaluate', 'visualize'],
                        help='Mode to run: infer | finetune | evaluate | visualize')
    parser.add_argument('--model', type=str, default='ViT-B/32', help='CLIP model variant')
    parser.add_argument('--dataset', type=str, default='/kaggle/input/superstition-dataset/Big Data', help='Path to dataset root')
    parser.add_argument('--model_path', type=str, default='https://huggingface.co/Mohammad121/Finetuned_CLIP-32_no_superstition/resolve/main/fine_tuned_model.pt', 
                        help='Hugging Face path to fine-tuned model')
    parser.add_argument('--csv', type=str, default='output/clip_superstition_dataset.csv', help='Path to output CSV')
    parser.add_argument('--save_dir', type=str, default='output/', help='Output save directory')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'infer':
        clip_inference.run_inference(args.dataset, args.model, args.save_dir)

    elif args.mode == 'finetune':
        fine_tuning.run_kfold_training(
            csv_path=args.csv,
            model_name=args.model,
            save_path=args.save_dir
        )

    elif args.mode == 'evaluate':
        clip_inference.run_finetuned_evaluation(args.dataset, args.model_path, args.save_dir)

    elif args.mode == 'visualize':
        visualize.run_visualization()

    else:
        print("Invalid mode.")
