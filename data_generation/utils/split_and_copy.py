import shutil
import random

def split_and_move(src_dir, train_dst, val_dst, val_count=20):
    print(f"\n>>> Task: Splitting Dataset -> {val_count} images to Val, rest to Train")
    
    # Ensure destination directories exist
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    all_files = list(src_dir.glob("*.jpg"))
    random.shuffle(all_files)

    # Split files
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    
    # Copy files
    for f in val_files:
        shutil.move(f, val_dst / f.name)
    
    for f in train_files:
        shutil.move(f, train_dst / f.name)

    print(f"Split complete: {len(train_files)} training images, {len(val_files)} validation images.")
