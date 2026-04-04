import cv2
import albumentations as A
from pathlib import Path
from aps360_lpr.data_generation.pipeline import (
    domain_randomizer_dir,
    clean_plate_out,
    domain_random_output_dir,
)

INPUT_DIR = clean_plate_out
OUTPUT_DIR = domain_random_output_dir

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

aging_pipeline = A.Compose(
    [
        A.Perspective(scale=(0.02, 0.07), keep_size=True, p=0.5),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 9), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Defocus(radius=(3, 5), alias_blur=(0.1, 0.5), p=1.0),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0),
            ],
            p=0.5,
        ),
        A.ImageCompression(quality_lower=30, quality_upper=80, p=0.6),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=1.0
                ),
                A.RandomShadow(
                    num_shadows_lower=1,
                    num_shadows_upper=1,
                    shadow_dimension=4,
                    shadow_roi=(0, 0, 1, 1),
                    p=1.0,
                ),
                A.RandomSunFlare(
                    flare_roi=(0.2, 0.2, 0.8, 0.8),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=1,
                    src_radius=30,
                    src_color=(255, 255, 255),
                    p=0.3,
                ),
            ],
            p=0.5,
        ),
        A.PixelDropout(dropout_prob=0.005, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=4, sat_shift_limit=15, val_shift_limit=10, p=0.2
        ),
    ]
)


def process_plates():
    input_path = Path(INPUT_DIR)
    # Get all .jpg files in the directory
    image_files = list(input_path.glob('*.jpg'))

    if not image_files:
        print(f'No .jpg files found in {INPUT_DIR}.')
        return

    print(f'Found {len(image_files)} plates. Starting physical aging process...')

    success_count = 0

    for img_path in image_files:
        # 1. Read the image
        # Note: OpenCV reads images in BGR format by default
        image = cv2.imread(str(img_path))
        if image is None:
            print(f'Warning: Could not read image {img_path.name}')
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
            print(f'Processed {success_count} / {len(image_files)} plates...')

    print(
        f"Done! Successfully aged {success_count} plates. Saved to the '{OUTPUT_DIR}' directory."
    )


if __name__ == '__main__':
    process_plates()
