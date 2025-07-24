import cv2
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python tim.py <input_image_path> <output_folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = cv2.imread(input_path)
    if img is None:
        print("Could not load image.")
        sys.exit(1)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define target color and range
    target_color = np.array([218, 203, 226])  # RGB
    tolerance = 30  # +/- range for matching

    lower = np.clip(target_color - tolerance, 0, 255)
    upper = np.clip(target_color + tolerance, 0, 255)

    # Create mask
    mask = cv2.inRange(img_rgb, lower, upper)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop and save each found contour
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip very small regions (noise)
        if w < 30 or h < 30:
            continue

        cropped = img[y:y+h, x:x+w]
        out_path = os.path.join(output_folder, f"paper_crop_{i}.jpg")
        cv2.imwrite(out_path, cropped)
        print(f"Saved: {out_path}")

        # Optional: Draw on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Detected Area", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
