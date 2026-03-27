from pathlib import Path
import cv2
import pytesseract
import imutils
import numpy as np


def recognize_license_plate(image_path):
    # 1. Load the image
    print(f'DEBUG: Processing {image_path}')
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print('Error: Could not load image.')
        return None

    # 2. Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Using Gaussian Blur which might retain more detail for small images compared to strong Bilateral
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Apply edge detection to identify structural boundaries
    edged = cv2.Canny(blurred, 30, 200)

    # NEW: Dilate/Close the edges to close gaps in the contour
    # key: Use MORPH_CLOSE to connect broken ridges (plate borders)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 4. Contour recognition
    # Find contours and sort them from largest to smallest, keeping only the top 30
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    print(f'DEBUG: Found {len(contours)} top contours (processing max 30).')

    location = None
    plate_rect = None

    for i, contour in enumerate(contours):
        # Filter out small contours
        area = cv2.contourArea(contour)
        if area < 200:
            continue

        # Get bounding box properties
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        rect_area = w * h
        extent = area / float(rect_area)

        print(
            f'DEBUG: Contour {i} - area: {area:.1f}, aspect: {aspect_ratio:.2f}, extent: {extent:.2f}'
        )

        # Check for license plate generic properties
        # Aspect ratio: 1.5 - 5.0 (Covers square to wide)
        # Extent: Rectangular shapes fill their bounding box (> 0.5 usually)
        # Relaxed checks
        if 1.2 <= aspect_ratio <= 5.0 and extent > 0.45:
            location = contour
            plate_rect = (x, y, w, h)
            print(f'DEBUG: Contour {i} accepted as potential plate.')
            break

    if location is None:
        print('Could not detect a license plate contour.')
        # Debug: Save the edged image to see what's happening
        debug_output_path = f'debug_edged_{Path(image_path).name}'
        cv2.imwrite(debug_output_path, edged)
        return None

    # 6. Crop the image to the rectangular shape
    # Use the bounding rect with padding
    x, y, w, h = plate_rect
    pad = 5
    # Ensure padding doesn't go out of bounds
    img_h, img_w = gray.shape
    x_pad = max(0, x - pad)
    y_pad = max(0, y - pad)
    w_pad = min(img_w - x_pad, w + 2 * pad)
    h_pad = min(img_h - y_pad, h + 2 * pad)

    cropped_plate = gray[y_pad : y_pad + h_pad, x_pad : x_pad + w_pad]

    # Preprocess crop for better OCR
    # Resize crop if it's too small (Tesseract likes characters > 30px height)
    if cropped_plate.shape[0] < 50:
        scale = 50.0 / cropped_plate.shape[0]
        cropped_plate = cv2.resize(
            cropped_plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    # Thresholding to make text stand out (Adaptive or Otsu)
    cropped_plate = cv2.threshold(
        cropped_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Save debug crop
    debug_crop_path = f'debug_crop_{Path(image_path).name}'
    cv2.imwrite(debug_crop_path, cropped_plate)
    print(f'DEBUG: Saved cropped plate to {debug_crop_path}')

    # 7. Use Tesseract OCR to recognize characters
    # Try different PSM modes and Inversion
    configs = [
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    ]

    best_text = ''
    # Try with normal cropped image and inverted image (for dark text on light background vs light on dark)
    for img_variant in [cropped_plate, cv2.bitwise_not(cropped_plate)]:
        for config in configs:
            text = pytesseract.image_to_string(img_variant, config=config).strip()
            print(f"DEBUG: OCR Attempt: '{text}'")
            # Heuristic: Prefer longer strings that look like plates
            if len(text) > len(best_text):
                best_text = text
            # If we match a pattern (e.g. 3-4 letters + 3-4 numbers), maybe stop?
            # For now just max length.

    return best_text


# Example usage:
# result = recognize_license_plate("path_to_ontario_car_image.jpg")
# print("Recognized Plate:", result)
