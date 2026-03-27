import cv2
import pytesseract


def extract_plate_text_ocr(cropped_image_path):
    """
    Extracts text from a pre-cropped license plate image using Tesseract OCR.
    """
    # 1. Load the pre-cropped image
    cropped_plate = cv2.imread(cropped_image_path)
    if cropped_plate is None:
        print(f'Error: Could not load image at {cropped_image_path}.')
        return None

    # 2. Convert to grayscale
    # Tesseract generally performs better on single-channel (grayscale) images
    gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

    # Optional: You can uncomment the line below to apply Otsu's thresholding
    # if your cropped plates have low contrast.
    # _, gray_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Configure and run Tesseract OCR
    # --oem 3: Use the default OCR Engine Mode.
    # --psm 8: Page Segmentation Mode 8 treats the image as a single word.
    # -c tessedit_char_whitelist: Limits the output to standard plate characters.
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(gray_plate, config=custom_config)

    # 4. Return the cleaned string
    return text.strip()


# Example usage:
# ocr_result = extract_plate_text_ocr("path_to_cropped_plate_1.jpg")
# print("Baseline OCR Result:", ocr_result)
