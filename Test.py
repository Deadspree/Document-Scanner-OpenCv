import cv2

# Read an image
image = cv2.imread('1.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bitwise NOT operation
result = cv2.bitwise_not(gray_image)

# Display the original and result images
cv2.imshow('Original Image', gray_image)
cv2.imshow('Bitwise NOT Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()