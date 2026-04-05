import cv2
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

def embed_plate(plate_img, bg_img, coordinates):
    h, w = plate_img.shape[:2]
    
    # Source points (clean plate image corners)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Destination points from crop_points.json
    # Points in JSON are typically:
    # 1: Top-Left, 2: Top-Right, 3: Bottom-Left, 4: Bottom-Right
    # OpenCV expects: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    try:
        dst_pts = np.float32([
            coordinates["1"],
            coordinates["2"],
            coordinates["4"],
            coordinates["3"]
        ])
    except KeyError:
        return bg_img
        
    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp the plate image to fit the background perspective
    bg_h, bg_w = bg_img.shape[:2]
    warped_plate = cv2.warpPerspective(plate_img, M, (bg_w, bg_h))
    
    # Create a mask for the warped plate
    mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    
    # Erode the mask slightly so the blur happens more inside the plate boundaries
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 3)
    
    # Blur the mask to create a soft edge (feathering)
    # CHANGE THESE PARAMETERS to adjust the blur amount. The numbers must be ODD (e.g., 31, 51, 75, 101).
    # Since your images are high resolution (3000px+), you need a large kernel size to see the effect.
    blur_kernel_size = (21, 21)
    blurred_mask = cv2.GaussianBlur(mask, blur_kernel_size, 0)
    
    # Normalize mask to 0-1 for alpha blending and extend to 3 channels
    alpha = blurred_mask.astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=-1)
    
    # Perform alpha blending
    bg_float = bg_img.astype(float)
    plate_float = warped_plate.astype(float)
    
    result_float = bg_float * (1.0 - alpha) + plate_float * alpha
    result = result_float.astype(np.uint8)
    
    return result

def main():
    base_dir = Path(__file__).parent.resolve()
    
    # Navigate up to the repository root directory
    project_root = base_dir.parents[3]
    dataset_dir = project_root / 'dataset'
    
    clean_plates_dir = dataset_dir / 'temp' / 'domain_randomizer_out_'
    templates_dir = base_dir / 'background_templates'
    json_path = base_dir / 'crop_points.json'
    
    # Output directory
    output_dir = dataset_dir / 'background_embedder_out_'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load coordinates
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, 'r') as f:
        crop_points = json.load(f)
        
    # Create mapping from template name to coordinates
    template_map = {item['img_name']: item['coordinates'] for item in crop_points}
    template_names = list(template_map.keys())
    
    if not template_names:
        print("Error: No templates found in crop_points.json.")
        return
        
    # Pre-load all template images to speed up processing
    print("Loading templates into memory...")
    loaded_templates = {}
    for img_name in template_names:
        img_path = templates_dir / img_name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                loaded_templates[img_name] = img
        else:
            print(f"Warning: Template image {img_name} not found in {templates_dir}")
                
    if not loaded_templates:
        print("Error: No template images could be loaded.")
        return
        
    # Get all clean plate image paths
    plate_paths = list(clean_plates_dir.glob('*.jpg')) + list(clean_plates_dir.glob('*.png'))
    if not plate_paths:
        print(f"Error: No clean plates found in {clean_plates_dir}")
        return
        
    print(f"Found {len(plate_paths)} clean plates and {len(loaded_templates)} valid templates.")
    print(f"Outputting embedded plates to {output_dir}...")
    
    # Main processing loop
    success_count = 0
    for i, plate_path in enumerate(tqdm(plate_paths, desc="Embedding plates")):
        plate_img = cv2.imread(str(plate_path))
        if plate_img is None:
            continue
            
        # Select a random backgound template
        template_name = random.choice(list(loaded_templates.keys()))
        bg_img = loaded_templates[template_name].copy()
        coords = template_map[template_name]
        
        # Embed the plate into the background
        result_img = embed_plate(plate_img, bg_img, coords)
        
        # Save output image with coordinates in the filename
        # Coordinates: 1=TL, 2=TR, 3=BL, 4=BR
        c1 = coords["1"]
        c2 = coords["2"]
        c3 = coords["3"]
        c4 = coords["4"]
        
        # Format string: x1_y1_x2_y2_x3_y3_x4_y4
        coord_str = f"{c1[0]:.1f}_{c1[1]:.1f}_{c2[0]:.1f}_{c2[1]:.1f}_{c3[0]:.1f}_{c3[1]:.1f}_{c4[0]:.1f}_{c4[1]:.1f}"
        
        # Include original plate name in filename
        original_name = plate_path.stem
        out_filename = f"embedded_{coord_str}_{original_name}_{i}.jpg"
        out_path = output_dir / out_filename
        cv2.imwrite(str(out_path), result_img)
        success_count += 1
        
    print(f"Processing complete! successfully generated {success_count} images.")

if __name__ == '__main__':
    main()
