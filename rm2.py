from PIL import Image
from rembg import remove
from io import BytesIO

def remove_bg(image: Image.Image) -> Image.Image:
    """
    Remove the background from a PIL Image and return an RGBA image.
    Converts the PIL Image to PNG bytes before passing to rembg.remove().
    """
    # Ensure the input image is in a mode suitable for rembg (RGB or RGBA)
    # If the input is something like 'L' (grayscale), rembg might struggle without conversion.
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    # Convert the PIL Image to bytes in a recognized format (e.g., PNG)
    img_byte_arr = BytesIO()
    # Always save as PNG if you intend to preserve transparency
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Pass the encoded image bytes to rembg.remove()
    removed_bytes = remove(img_bytes)

    # Open the returned bytes as a PIL Image and ensure it's RGBA
    return Image.open(BytesIO(removed_bytes)).convert("RGBA")

def autocrop_transparent_borders(image: Image.Image) -> Image.Image:
    """
    Automatically crops the image to the bounding box of its non-transparent pixels.
    Assumes the input image is in RGBA mode.
    """
    if image.mode != 'RGBA':
        raise ValueError("Image must be in RGBA mode for autocrop_transparent_borders.")

    # Get the bounding box of the non-transparent region
    # getbbox() returns (left, upper, right, lower) of the bounding box.
    # It finds the smallest bounding box containing all non-zero pixels in the alpha channel.
    bbox = image.getbbox()

    if bbox: # If a non-empty bounding box is found
        return image.crop(bbox)
    else:
        # If bbox is None, it means the image is entirely transparent or empty
        # In this case, return a completely transparent image of minimal size, or the original
        # depending on desired behavior. For simplicity, we'll return an empty 1x1 transparent image.
        print("Warning: Image is entirely transparent after background removal. Returning a 1x1 transparent image.")
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)) # 1x1 completely transparent pixel

def process_image_auto_crop(input_path: str, output_path: str):
    """
    Remove the background, then automatically crop the transparent borders.
    """
    original = Image.open(input_path).convert("RGBA") # Ensure original is RGBA for consistency

    print("Step 1: Removing background...")
    # Step 1: Remove the background
    bg_removed = remove_bg(original)

    print("Step 2: Automatically cropping transparent borders...")
    # Step 2: Auto-crop the transparent borders
    auto_cropped_image = autocrop_transparent_borders(bg_removed)

    # Step 3: Save the result
    # Always save as PNG to preserve transparency
    auto_cropped_image.save(output_path, format='PNG')

    print(f"Processed image (background removed and auto-cropped) saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    input_image_path = "test4.jpg"      # Replace with your image file
    output_image_path = "output_auto_cropped.png"

    try:
        process_image_auto_crop(input_image_path, output_image_path)
    except FileNotFoundError:
        print(f"Error: Input image '{input_image_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()