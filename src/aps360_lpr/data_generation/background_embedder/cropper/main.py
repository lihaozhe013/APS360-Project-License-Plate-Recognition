from pathlib import Path
from src.image_scanner import ImageScanner

base_dir = Path(__file__).parent.resolve()
input_dir = base_dir / 'background_templates'
output_dir = base_dir / 'output'

if __name__ == '__main__':
    app = ImageScanner(input_dir, output_dir)
    app.run()
