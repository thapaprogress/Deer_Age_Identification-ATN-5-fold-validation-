import os
from pathlib import Path
import json

# Dataset path
dataset_path = r"c:\Users\PRAJNA WORLD TECH\OneDrive\Desktop\atn\deer data"

# Analyze dataset structure
dataset_info = {
    "total_deer": 0,
    "age_distribution": {},
    "deer_details": [],
    "image_formats": set(),
    "total_images": 0
}

# Iterate through deer folders
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    
    if os.path.isdir(folder_path) and folder.startswith("Deer_id"):
        dataset_info["total_deer"] += 1
        
        # Extract age from folder name
        age = None
        if "age" in folder.lower():
            try:
                age_str = folder.split("age")[1].strip().replace(")", "").replace("(", "").strip()
                age = int(age_str.split()[0])
            except:
                age = "Unknown"
        
        # Count images in this deer folder
        deer_images = 0
        image_categories = {}
        
        # Check subdirectories
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                images_in_category = []
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(file)[1].lower()
                        dataset_info["image_formats"].add(ext)
                        images_in_category.append(file)
                        deer_images += 1
                        dataset_info["total_images"] += 1
                
                image_categories[subdir] = len(images_in_category)
        
        # Update age distribution
        if age:
            if age not in dataset_info["age_distribution"]:
                dataset_info["age_distribution"][age] = 0
            dataset_info["age_distribution"][age] += 1
        
        # Store deer details
        dataset_info["deer_details"].append({
            "folder": folder,
            "age": age,
            "total_images": deer_images,
            "categories": image_categories
        })

# Convert set to list for JSON serialization
dataset_info["image_formats"] = list(dataset_info["image_formats"])

# Sort age distribution
dataset_info["age_distribution"] = dict(sorted(dataset_info["age_distribution"].items()))

# Print summary
print("="*80)
print("DEER DATASET ANALYSIS")
print("="*80)
print(f"\nTotal Deer Individuals: {dataset_info['total_deer']}")
print(f"Total Images: {dataset_info['total_images']}")
print(f"Image Formats: {', '.join(dataset_info['image_formats'])}")
print(f"\nAge Distribution:")
for age, count in dataset_info["age_distribution"].items():
    print(f"  Age {age}: {count} deer")

print(f"\nAverage images per deer: {dataset_info['total_images'] / dataset_info['total_deer']:.2f}")

# Sample deer details
print(f"\nSample Deer Details (first 5):")
for deer in dataset_info["deer_details"][:5]:
    print(f"\n  {deer['folder']}")
    print(f"    Age: {deer['age']}")
    print(f"    Total Images: {deer['total_images']}")
    print(f"    Categories: {deer['categories']}")

# Save to JSON
with open("dataset_analysis.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

print(f"\n\nFull analysis saved to dataset_analysis.json")
