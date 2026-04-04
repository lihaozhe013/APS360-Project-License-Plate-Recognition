from pathlib import Path
from aps360_lpr.data_generation.clean_plate_generator.src.utils import generate_plate
from aps360_lpr.data_generation.clean_plate_generator.src.utils import init_out
from jinja2 import Template
from playwright.sync_api import sync_playwright
from aps360_lpr.data_generation.pipeline import (
    base_dir,
    dataset_base_dir,
    scripts_base_dir,
    clean_plate_out,
    clean_plate_generator_dir,
    config
)

num_of_plates = config.num_of_plates
out_path = clean_plate_out

template_html_path = clean_plate_generator_dir / 'src' / 'template.html'


def generate_images():
    with open(template_html_path, 'r') as file:
        template_content = file.read()

    template = Template(template_content)

    print('Starting generation...')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for i in range(num_of_plates):
            plate_letter, plate_num = generate_plate(4, 3)
            # Use letters and numbers directly for template
            L = plate_letter
            D = plate_num

            # Construct full plate string for filename
            full_plate = f'{L}-{D}'

            rendered_content = template.render(L=L, D=D, status='Awesome')
            output_html_filename = f'{full_plate}.html'
            output_html_path = out_path / output_html_filename

            with open(output_html_path, 'w') as file:
                file.write(rendered_content)

            output_jpg_path = out_path / f'{full_plate}.jpg'

            # process paths for playwright (absolute path is safer)
            absolute_html_path = output_html_path.resolve()
            page.goto(f'file://{absolute_html_path}')

            # Take screenshot of the plate element
            # Using the ID from template.html: <div class="plate-container" id="plate">
            page.locator('#plate').screenshot(path=output_jpg_path)

            print(f'Plate {full_plate} successfully generated!')

        browser.close()


if __name__ == '__main__':
    init_out(clean_plate_generator_dir, out_path)
    generate_images()
