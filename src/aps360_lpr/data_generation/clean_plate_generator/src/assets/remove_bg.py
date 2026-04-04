from PIL import Image
import os


def remove_white_background(input_path, output_path, tolerance=220):
    """
    Reads an image, finds all white (or near-white) pixels,
    and makes them transparent. Saves the result as a PNG.
    """
    # 1. Open image and convert to RGBA (Red, Green, Blue, Alpha)
    img = Image.open(input_path).convert('RGBA')
    datas = img.getdata()

    new_data = []
    for item in datas:
        # item is a tuple: (R, G, B, A)
        # Check if the pixel is white or near-white based on the tolerance
        if item[0] > tolerance and item[1] > tolerance and item[2] > tolerance:
            # Change the alpha value to 0 (completely transparent)
            # Keeping the RGB as white (255, 255, 255) ensures clean edges
            new_data.append((255, 255, 255, 0))
        else:
            # Keep the original pixel
            new_data.append(item)

    # 2. Update the image with the new transparent pixels
    img.putdata(new_data)

    # 3. Save as PNG (JPG does not support transparency)
    img.save(output_path, 'PNG')
    print(f'Saved transparent image to: {output_path}')


if __name__ == '__main__':
    # Define your file paths
    # Note: Adjust these if your folder structure is different
    ontario_input = 'Ontario.jpg'
    ontario_output = 'Ontario_transparent.png'

    crown_input = 'crown.png'
    crown_output = 'crown_transparent.png'

    discover_input = 'yours_to_discover.jpg'
    discover_output = 'yours_to_discover_transparent.png'

    # Process all images
    # A tolerance of 220 catches JPG compression artifacts
    print('Processing images...')

    if os.path.exists(ontario_input):
        remove_white_background(ontario_input, ontario_output)

    if os.path.exists(crown_input):
        remove_white_background(crown_input, crown_output)

    if os.path.exists(discover_input):
        remove_white_background(discover_input, discover_output)

    print('Done! Update your HTML to use the new .png files.')
