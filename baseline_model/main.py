from baseline import recognize_license_plate
from orc import extract_plate_text_ocr
from pathlib import Path

base_dir = Path(__file__).parent.resolve()
test_orc_dir = base_dir / "test_orc"
test_contour_orc_dir = base_dir / "test_contour_orc"

if __name__ == "__main__":
    print(f"======= Testing ORC =======")
    for img_dir in test_orc_dir.iterdir():
        result = extract_plate_text_ocr(img_dir)
        print(f"Testing: {img_dir}")
        print(f"{result}")

    print(f"===== Testing Contour Detection + ORC =====")
    for img_dir in test_contour_orc_dir.iterdir():
        result = recognize_license_plate(img_dir)
        print(f"Testing: {img_dir}")
        print(f"{result}")
