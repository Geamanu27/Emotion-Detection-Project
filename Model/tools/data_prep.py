import os
import shutil
from tqdm import tqdm  # Progress bar

# CONFIGURATION
# --------------------------
# Path to your current 7-class dataset (Update if on Colab!)
original_dataset_dir = '/content/dataset'
# Where to create the new 3-class dataset
new_dataset_dir = '/content/dataset_3class'

# DEFINE THE GROUPING
# --------------------------
class_mapping = {
    'angry': 'Negative',
    'disgust': 'Negative',
    'fear': 'Negative',
    'sad': 'Negative',
    'happy': 'Positive',
    'surprise': 'Positive',
    'neutral': 'Neutral'
}


def create_grouped_dataset(subset):
    """
    subset: 'train' or 'validation'
    """
    print(f"\nProcessing {subset} set...")
    source_subset_dir = os.path.join(original_dataset_dir, subset)
    dest_subset_dir = os.path.join(new_dataset_dir, subset)

    # Create 3 target folders (Positive, Negative, Neutral)
    for category in ['Positive', 'Negative', 'Neutral']:
        os.makedirs(os.path.join(dest_subset_dir, category), exist_ok=True)

    # Loop through original 7 folders
    for emotion_folder in os.listdir(source_subset_dir):
        emotion_path = os.path.join(source_subset_dir, emotion_folder)

        # Skip if it's not a folder
        if not os.path.isdir(emotion_path):
            continue

        # Determine target group
        if emotion_folder in class_mapping:
            target_group = class_mapping[emotion_folder]
            target_path = os.path.join(dest_subset_dir, target_group)

            # Copy files
            files = os.listdir(emotion_path)
            for file in tqdm(files, desc=f"Copying {emotion_folder} -> {target_group}"):
                src = os.path.join(emotion_path, file)
                dst = os.path.join(target_path, file)
                shutil.copy(src, dst)


# Run it
create_grouped_dataset('train')
create_grouped_dataset('validation')  # or 'val' depending on your folder name

print("\n✅ Dataset grouped successfully in /dataset_3class")