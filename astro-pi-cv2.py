import cv2
import numpy as np
import time
import os
from picamzero import Camera
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

# Constants
PIXEL_TO_KM_CONVERSION = 0.12648  # Conversion factor from pixels/sec to km/sec
DEFAULT_DELTA_T = 1  # Time difference in seconds (assumed to be 1 second interval)

# Initialize camera
cam = Camera()
cam.start_preview()

# Attempt to capture 10 images
cam.capture_sequence(filename="sequence", num_images=10, interval=1)
image_filenames = [f"sequence-{i:02d}" for i in range(1,11)]

# Function to get image capture time from EXIF data
def get_image_capture_time(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'DateTimeOriginal':
                    return value
    except Exception as e:
        print(f"Error reading EXIF data from {image_path}: {e}")
    return None

# Function to calculate motion and speed
def compute_speed(image1, image2, delta_t):
    # Convert to grayscale
    image1_gray = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    image2_gray = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Create OFAST detector and detect keypoints
    fast = cv2.FastFeatureDetector_create()
    keypoints1 = fast.detect(image1_gray, None)
    keypoints2 = fast.detect(image2_gray, None)

    # Create ORB descriptor to compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.compute(image1_gray, keypoints1)
    keypoints2, descriptors2 = orb.compute(image2_gray, keypoints2)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Calculate displacement vectors
    displacements = []
    for match in matches:
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt
        displacement = np.linalg.norm(np.array(kp2) - np.array(kp1))
        displacements.append(displacement)

    # Calculate the average displacement
    avg_displacement = np.mean(displacements)
    # Estimate speed in pixels per second
    speed_pixels_per_second = avg_displacement / delta_t

    # Convert to kilometers per second
    speed_kilometers_per_second = speed_pixels_per_second * PIXEL_TO_KM_CONVERSION

    return speed_kilometers_per_second

# List to store image paths
image_paths = [f"{filename}.jpg" for filename in image_filenames]

# Main computation for speed across image pairs
speeds = []
for i in range(0, len(image_paths) - 1, 2):  # Process pairs of images
    image1 = image_paths[i]
    image2 = image_paths[i + 1]

    # Get capture times for delta_t (can be adjusted if EXIF data available)
    capture_time1 = get_image_capture_time(image1)
    capture_time2 = get_image_capture_time(image2)

    if capture_time1 is None or capture_time2 is None:
        delta_t = DEFAULT_DELTA_T
    else:
        time_format = "%Y:%m:%d %H:%M:%S"
        time1 = datetime.strptime(capture_time1, time_format)
        time2 = datetime.strptime(capture_time2, time_format)
        delta_t = (time2 - time1).total_seconds()

    # Calculate speed for this image pair
    speed = compute_speed(image1, image2, delta_t)
    speeds.append(speed)

# Calculate mean speed from all pairs
mean_speed = np.mean(speeds)

# Save results to a text file
output_filename = "ISS_speed_results.txt"
with open(output_filename, 'w') as f:
    f.write(f"Mean Speed of the ISS: {mean_speed:.2f} km/s\n")

print(f"Mean speed of the ISS: {mean_speed:.2f} km/s")
print(f"Results saved in {output_filename}")
