import cv2
import numpy as np

# List of images to process (file name, label)
images = [
    ("slamdunk22.jpg", "good"),
    ("bad.jpg", "bad")
]

for path, name in images:
    # Load the image
    img = cv2.imread(path)

    # Check if the image is loaded correctly
    if img is None:
        print(f"Error: Cannot load {path}")
        continue

    # 1. Apply bilateral filter to smooth colors while preserving edges
    color = cv2.bilateralFilter(img, d=7, sigmaColor=100, sigmaSpace=100)

    # 2. Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply median blur to reduce noise
    gray = cv2.medianBlur(gray, 5)

    # 4. Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        5
    )

    # 5. Combine the color image with the edge mask
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Resize the result to match the original image size (safety step)
    cartoon = cv2.resize(cartoon, (img.shape[1], img.shape[0]))

    # Save the cartoon image
    cv2.imwrite(f'cartoon_{name}.jpg', cartoon)

    # Create a side-by-side comparison image
    combined = np.hstack((img, cartoon))

    # Save the comparison image
    cv2.imwrite(f'compare_{name}.jpg', combined)

    print(f"{name} image processed and saved!")

# Display the last processed result (optional)
cv2.imshow("Last Result", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
