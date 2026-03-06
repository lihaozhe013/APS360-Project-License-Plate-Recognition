import os
import cv2
import albumentations as A
from pathlib import Path

script_path = Path(__file__).resolve()
root_dir = script_path.parent

INPUT_DIR = root_dir / 'clean_plates'
OUTPUT_DIR = root_dir / 'aged_plates'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Define Domain Randomization Pipeline (Local Aging)
# ==========================================
# Note: The parameters here are relatively conservative, 
# as global processing (blur/compression) will be applied later.
aging_pipeline = A.Compose([
    # 1. Simulate material aging: yellowing, fading, uneven exposure
    # Use ColorJitter to slightly perturb brightness, contrast, and saturation
    # A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.05, p=0.8),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.1, p=0.8),
    
    # Use RGBShift to simulate a slight color cast (e.g., white background turning yellowish)
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=10, p=0.5),
    
    # 2. Simulate physical stains: mud spots, splashes, chipped paint
    # CoarseDropout randomly generates small dark squares (simulating mud or large stains)
    # A.CoarseDropout(
    #     max_holes=2,             # Maximum number of stains
    #     max_height=4,           # Maximum height of a stain (pixels)
    #     max_width=4,            # Maximum width of a stain (pixels)
    #     min_holes=1,             # Minimum number of stains
    #     fill_value=(50, 60, 50), # Fill color: Dark grayish-brown (RGB format)
    #     p=0.4                    # 40% probability of applying these large stains
    # ),
    
    # PixelDropout randomly blacks out individual pixels, simulating fine dust or micro-scratches
    A.PixelDropout(dropout_prob=0.005, p=0.5) 
])

def process_plates():
    input_path = Path(INPUT_DIR)
    # Get all .jpg files in the directory
    image_files = list(input_path.glob("*.jpg"))
    
    if not image_files:
        print(f"No .jpg files found in {INPUT_DIR}.")
        return

    print(f"Found {len(image_files)} plates. Starting physical aging process...")

    success_count = 0
    
    for img_path in image_files:
        # 1. Read the image
        # Note: OpenCV reads images in BGR format by default
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path.name}")
            continue
            
        # 2. Color space conversion
        # Albumentations recommends using RGB format for color-related augmentations
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Apply the Albumentations pipeline
        augmented = aging_pipeline(image=image_rgb)
        image_aged_rgb = augmented['image']
        
        # 4. Convert back to BGR format for OpenCV to save correctly
        image_aged_bgr = cv2.cvtColor(image_aged_rgb, cv2.COLOR_RGB2BGR)
        
        # 5. Save the image (keeping the original filename, which is the plate number)
        output_filepath = Path(OUTPUT_DIR) / img_path.name
        cv2.imwrite(str(output_filepath), image_aged_bgr)
        
        success_count += 1
        
        # Simple progress printer
        if success_count % 1000 == 0:
            print(f"Processed {success_count} / {len(image_files)} plates...")

    print(f"Done! Successfully aged {success_count} plates. Saved to the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    process_plates()