import cv2
import numpy as np
import os

def order_points(pts):
    """
    Orders the four points of a rectangle in top-left, top-right, bottom-right, bottom-left order.
    This is crucial for correct perspective transformation.
    """
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second is the top-right, the third is the bottom-right,
    # and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum,
    # whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    """
    Applies a four-point perspective transform to an image.
    This function takes an image and the ordered four points of a quadrilateral
    and returns a "scanned" (top-down, bird's eye view) of the image.
    """
    # Obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped

def process_invoice(image_path):
    """
    Loads an image, detects the largest rectangular contour (assumed to be the invoice),
    applies a perspective transform, and then thresholds the image to remove the background.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load the image
    image = cv2.imread(image_path)

    # If image loading failed
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by their area in descending order and keep only the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Initialize the screen contour
    screenCnt = None

    # Loop over the contours
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our approximated contour has four points, then we can assume we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # If no 4-point contour was found, print a message and return
    if screenCnt is None:
        print("Could not find a clear rectangular contour (invoice) in the image.")
        # Optionally, you might want to return the original image or a grayscale version
        # return image # Or return None
        return

    # Apply the four point perspective transform to obtain a top-down view
    warped = four_point_transform(image, screenCnt.reshape(4, 2))

    # Convert the warped image to grayscale for thresholding
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image (black text on white background)
    # This is more robust for varying lighting conditions across the document.
    # ADAPTIVE_THRESH_GAUSSIAN_C calculates the threshold as a weighted sum of neighborhood values
    # blockSize is the size of a pixel neighborhood that is used to calculate a threshold value
    # C is a constant subtracted from the mean or weighted mean
    thresh = cv2.adaptiveThreshold(warped_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Display the original and processed images (for debugging/visualization)
    cv2.imshow("Original Image", image)
    cv2.imshow("Edged Image", edged)
    cv2.imshow("Scanned (Perspective Corrected)", warped)
    cv2.imshow("Processed (Background Removed)", thresh)

    # Save the processed image
    output_filename = "processed_invoice.jpg"
    cv2.imwrite(output_filename, thresh)
    print(f"Processed invoice saved as {output_filename}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your invoice image
# Make sure 'test4.jpg' is in the same directory as this script, or provide the full path.
image_file = 'test4.jpg'
process_invoice(image_file)
