

import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")

# Load your image
img = cv2.imread('./bricklaying_data/frame_0000.png')
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()


