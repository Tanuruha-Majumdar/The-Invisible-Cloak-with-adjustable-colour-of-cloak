import cv2
import numpy as np




def noise_removal(mask):
    cv2.imshow("mask", mask)
    # kernel=np.ones((10, 10), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=5)  # noise removal
    # mask = cv2.dilate(mask, kernel, iterations=8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)  # noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    return mask


def empty(x):
    pass


def hsv_masking():
    cv2.namedWindow("TRACKBARS")
    # cv2.resizeWindow("TRACKBARS", 650, 250)
    cv2.createTrackbar("Hue Min", "TRACKBARS", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TRACKBARS", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TRACKBARS", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TRACKBARS", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TRACKBARS", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TRACKBARS", 255, 255, empty)
    vid = cv2.VideoCapture(0)
    while vid.isOpened():
        success, img = vid.read()
        if not success:
            break
        img = cv2.resize(img, (900, 520))
        img = np.flip(img, 1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TRACKBARS")
        h_max = cv2.getTrackbarPos("Hue Max", "TRACKBARS")
        s_min = cv2.getTrackbarPos("Sat Min", "TRACKBARS")
        s_max = cv2.getTrackbarPos("Sat Max", "TRACKBARS")
        v_min = cv2.getTrackbarPos("Val Min", "TRACKBARS")
        v_max = cv2.getTrackbarPos("Val Max", "TRACKBARS")

        # print(h_min,h_max,s_min,s_max,v_min,v_max)
        lower = [h_min, s_min, v_min]
        upper = [h_max, s_max, v_max]
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        mask = noise_removal(mask)
        color_mask = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow("Original",img)
        # cv2.imshow("HSV",hsv_img)
        cv2.imshow("HSV_MASK", mask)
        cv2.imshow("colormask", color_mask)

        k = cv2.waitKey(1)
        if k == 27:  # esc ascii
            break

    vid.release()
    cv2.destroyAllWindows()
    return lower, upper
