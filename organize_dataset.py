import os
import shutil
from pathlib import Path
import random

def create_directories(base_path):
    """create necessary directories if they don't exist"""
    directories = [
        'data/train/stop_signs',
        'data/train/traffic_lights',
        'data/train/other',
        'data/test/stop_signs',
        'data/test/traffic_lights',
        'data/test/other'
    ]
    for dir_path in directories:
        Path(os.path.join(base_path, dir_path)).mkdir(parents=True, exist_ok=True)

def organize_dataset(temp_data_path, project_path, train_split=0.8):
    """organize GTSRB dataset into project structure"""
    # updated class mapping to include traffic lights
    class_mapping = {
        '14': 'stop_signs',      # stop signs
        '6': 'traffic_lights',   # traffic lights
        '7': 'traffic_lights',   # traffic lights (different type)
        '9': 'other',           # example of "other" class
        '19': 'other',          # example of "other" class
        '1': 'other'            # example of "other" class
    }
    
    # create directories
    create_directories(project_path)
    
    # path to the Train folder in the GTSRB dataset
    train_path = os.path.join(temp_data_path, 'Train')
    
    for class_id in os.listdir(train_path):
        class_path = os.path.join(train_path, class_id)
        if not os.path.isdir(class_path):
            continue
            
        # determine target category
        category = class_mapping.get(class_id, 'other')
        
        # get all images in the class
        images = [f for f in os.listdir(class_path) if f.endswith('.png')]
        random.shuffle(images)
        
        # split into train and test
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        print(f"Processing {category}: {len(train_images)} train images, {len(test_images)} test images")
        
        # copy images to appropriate directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(project_path, 'data', 'train', category, img)
            shutil.copy2(src, dst)
            
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(project_path, 'data', 'test', category, img)
            shutil.copy2(src, dst)
    
    print("Dataset organization complete!")

if __name__ == "__main__":
    # adjust these paths according to your setup
    TEMP_DATA_PATH = "temp_data"  # path to your unzipped dataset
    PROJECT_PATH = "."  # current directory
    
    organize_dataset(TEMP_DATA_PATH, PROJECT_PATH)
