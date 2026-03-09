import cv2
import pytesseract
import imutils
import numpy as np

def recognize_license_plate(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return None
    
    # 2. Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Using Bilateral Filter instead of standard Gaussian Blur as it keeps edges sharp while removing noise
    blurred = cv2.bilateralFilter(gray, 11, 17, 17) 

    # 3. Apply edge detection to identify structural boundaries
    edged = cv2.Canny(blurred, 30, 200) 

    # 4. Contour recognition
    # Find contours and sort them from largest to smallest, keeping only the top 10
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        
        # 5. Look for a rectangular shape (4 corners)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Ontario plates are roughly 12x6 inches, creating an aspect ratio of ~2.0
            # We allow a tolerance range (e.g., 1.5 to 2.7) to account for camera angles
            if 1.5 <= aspect_ratio <= 2.7:
                location = approx
                break

    if location is None:
        print("Could not detect a license plate contour.")
        return None

    # 6. Crop the image to the rectangular shape
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_plate = gray[x1:x2+1, y1:y2+1]

    # 7. Use Tesseract OCR to recognize characters
    # --psm 8: Treat the image as a single word.
    # -c tessedit_char_whitelist: Restrict to alphanumeric characters common on plates.
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(cropped_plate, config=custom_config)

    return text.strip()

# Example usage:
# result = recognize_license_plate("path_to_ontario_car_image.jpg")
# print("Recognized Plate:", result)