from PIL import Image
from rembg import remove
from io import BytesIO

def remove_bg(image: Image.Image) -> Image.Image:
    """
    Remove the background and return an RGBA image.
    Converts the PIL Image to PNG bytes before passing to rembg.remove().
    """
    # Convert the PIL Image to bytes in a recognized format (e.g., PNG)
    # This is the crucial step to get 'bytes-like object' that rembg expects.
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Pass the encoded image bytes to rembg.remove()
    removed_bytes = remove(img_bytes)

    # Open the returned bytes as a PIL Image
    return Image.open(BytesIO(removed_bytes)).convert("RGBA")

def process_image_no_crop(input_path: str, output_path: str):
    """
    Remove the background from the image without cropping.
    """
    # Open the image and ensure it's in RGBA mode for transparency handling
    original = Image.open(input_path).convert("RGBA")

    # Step 1: Remove the background (no cropping)
    bg_removed_image = remove_bg(original)

    # Step 2: Save the result
    bg_removed_image.save(output_path)

    print(f"Processed image (background removed, no crop) saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    input_image_path = "test4.jpg"  # Replace with your image file
    output_image_path = "output_no_crop.png"

    try:
        process_image_no_crop(input_image_path, output_image_path)
    except FileNotFoundError:
        print(f"Error: Input image '{input_image_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")