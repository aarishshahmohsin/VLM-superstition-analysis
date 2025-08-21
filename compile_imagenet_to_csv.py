import os
import pandas as pd

# Root directory (change this to your folder path)
root_dir = "/mnt/c/Users/HP/Downloads/archive"   # put your path here, e.g. "D:/dataset/"

# Supported image extensions
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

# Collect data
data = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_exts:
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)  # last folder name
            data.append({"folder": folder_name, "file": file, "path": file_path})

# Create DataFrame
df = pd.DataFrame(data)

print(df.head())
print(f"\nTotal images: {len(df)}")

# Optionally save to CSV
df.to_csv("image_dataset.csv", index=False)
