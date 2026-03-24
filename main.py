import cv2
import numpy as np

img = cv2.imread('slamdunk22.jpg')

# 1. Smooth the image while preserving edges (cartoon effect)
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

# 5. Combine color image with edge mask
cartoon = cv2.bitwise_and(color, color, mask=edges)

# Resize (safety step to match dimensions)
cartoon = cv2.resize(cartoon, (img.shape[1], img.shape[0]))

# Save the cartoon result image
cv2.imwrite('cartoon_result.jpg', cartoon)

# comparison image
combined = np.hstack((img, cartoon))
cv2.imwrite('compare.jpg', combined)

# Display original and cartoon images
cv2.imshow("Original", img)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()