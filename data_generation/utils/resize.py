import cv2
from pathlib import Path

def resize_data(directory):
    target_size = (300, 150) # width, height

    # Convert to Path object to handle various input types
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return

    print(f"Starting to resize images in {dir_path}...")
    
    # recursively find all .jpg files
    # rglob is case-sensitive on Linux, so we might miss .JPG if we only look for .jpg
    # to be safe, we can iterate all files and check suffix
    
    count = 0
    for file_path in dir_path.rglob('*'):
        if file_path.suffix.lower() == '.jpg':
            try:
                # cv2.imread needs string path
                str_path = str(file_path)
                img = cv2.imread(str_path)
                
                if img is None:
                    print(f"Warning: Could not read image {str_path}")
                    continue
                
                # Resize to 300x150, ignoring aspect ratio (stretch)
                # interpolation=cv2.INTER_AREA is generally good for shrinking, 
                # but for general resizing INTER_LINEAR is default. 
                # Since we might be upscaling or downscaling, linear is fine or area for downscale.
                # Let's use linear as a safe default for both directions or area if purely downsampling.
                # Given plates might be small or large, simple resize is fine.
                resized_img = cv2.resize(img, target_size)
                
                cv2.imwrite(str_path, resized_img)
                count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Finished resizing {count} images.")