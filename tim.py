from autocrop import cropper
import cv2
import random
import numpy as np

def remove_background(img, bbox, blur_radius=21, threshold=10):
    """Remove background outside the bounding box with smooth transition"""
    # Create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
    
    # Apply Gaussian blur to mask edges for smooth transition
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    
    # Convert mask to float and normalize
    mask = mask.astype(np.float32)/255.0
    
    # Create background with white or any other color
    background = np.ones_like(img, dtype=np.float32) * 255
    
    # Blend image with background using the mask
    img_float = img.astype(np.float32)
    result = img_float * mask[..., np.newaxis] + background * (1 - mask[..., np.newaxis])
    
    return result.astype(np.uint8)

# create autocropper with more efficient settings
autocropper = cropper.AutoCropper(model='mobilenetv2',
                                 cuda=True,
                                 use_face_detector=True)

# Load image with better color handling
img = cv2.imread('imgs/test3.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # More efficient than manual conversion

# get crop results with more controlled parameters
crop_ret = autocropper.crop(img_rgb,
                           topK=3,
                           crop_height=5,
                           crop_width=3,
                           filter_face=True,
                           single_face_center=True)

# Process and save each crop with background removal
for idx, bbox in enumerate(crop_ret):
    # Crop the image
    cropped_img = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    
    # Remove background with smooth edges
    no_bg_img = remove_background(img, bbox)
    cropped_no_bg = no_bg_img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    
    # Save both versions
    cv2.imwrite(f'cropped_{idx}.jpg', cropped_img)
    cv2.imwrite(f'cropped_no_bg_{idx}.jpg', cropped_no_bg)
    
    # Draw rectangles with consistent colors
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

# Display results with better window management
cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
cv2.imshow('Detection Results', img)
cv2.waitKey(0)
cv2.destroyAllWindows()