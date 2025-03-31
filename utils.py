
import cv2
import numpy as np
from imutils.perspective import four_point_transform

def doc_within_border(contours: list, start_point: tuple = (161, 15), endpoint: tuple = (479, 465)) -> bool:
    """
    Check that the contours (document) are within the given borders as rendered.
    Params: contours[list], start_point[tuple], endpoint[tuple]
    """
    x_min, y_min = start_point
    x_max, y_max = endpoint
    # Get the bounding box of the document contour (if it's a rectangle)
    x, y, w, h = cv2.boundingRect(contours)

    # Check if the bounding box is completely inside the defined rectangle
    if x >= x_min and y >= y_min and (x + w) <= x_max and (y + h) <= y_max:
        return True
    return False

def scan_detection(image):
    """
    Detects contours (border-lines) in a frame and returns coordinates
    """
    global document_contour
    WIDTH, HEIGHT, channels = image.shape

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    image_copy = image.copy()
    # Convert image to grayscale, then blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    # Convert contour to list of [x, y] coordinates
    # cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)
    return document_contour

scale = 0.2
def get_warped_iamge(image, scale=0.2):
    frame = scan_detection(image)
    frame_copy = image.copy()
    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    resulting_frame = cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0])))
    shape = resulting_frame.shape
    if shape[0] != 719 or shape[1] != 491:
        resulting_frame = cv2.resize(resulting_frame, (491, 719))
    return resulting_frame

