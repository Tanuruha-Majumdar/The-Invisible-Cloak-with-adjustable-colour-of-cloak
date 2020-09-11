import cv2
import numpy as np
from hsv_masking import hsv_masking, noise_removal

vid_cap = cv2.VideoCapture(0)

# Background Frame Capture

bg = 0
print("Press any key once background is ready to be captured")
cv2.waitKey(1000)
while cv2.waitKey(1) == -1:
    cam_working, bg = vid_cap.read()
    bg = cv2.flip(bg, 1)
    bg = cv2.resize(bg, (900, 520))
    cv2.imshow("Current Frame", bg)
cv2.imwrite("Background.jpg", bg)
cv2.destroyAllWindows()
print("\nBackground Frame successfully set")

# HSV mask for cloak

c=input("Press any key to start calliberation")
LOWER, UPPER = hsv_masking()
print(LOWER, UPPER)

# The magical implementation

c = input("Press any key to start the magic")
vid_cap.release()
background = cv2.imread("background.jpg")
vid_cap=cv2.VideoCapture(0)
while cv2.waitKey(1) == -1:
    works, pic = vid_cap.read()
    pic = cv2.flip(pic, 1)
    pic = cv2.resize(pic, (900, 520))
    #background = cv2.resize(background, (900, 520))
    hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(LOWER), np.array(UPPER))
    # mask = noise_removal(mask)
    pic = np.array(pic)
    res1 = cv2.bitwise_and(background, background, mask=mask)
    mask = cv2.bitwise_not(mask)  # everything other than the cloak

      # used for segmentation of color
    pic = cv2.bitwise_and(pic, pic, mask=mask)  # substituting the cloak
    final = cv2.add(res1, pic)
    cv2.imshow("Magic", final)

vid_cap.release()
cv2.destroyAllWindows()
