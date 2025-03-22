from flask import Flask, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import json
from imutils.perspective import four_point_transform


def scan_detection(image):
    """
    Detects contours (border-lines) in a frame and returns coordinates
    """
    WIDTH, HEIGHT, channels = image.shape

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    image_copy = image.copy()
    # Convert image to grayscale, then blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


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
    return cv2.drawContours(image_copy, [document_contour], -1, (0, 255, 0), 3),document_contour.reshape(4, 2)


app = Flask(__name__)

app.config['SECRET_KEY'] = "SOME_SECRETE_KEY12345"

socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/scrap')
def scrap():
    return json.dumps({"message":"Dumped.."})

@socketio.on('message')
def handle_message(msg):

    if isinstance(msg, dict) and 'image' in msg:
        file_bytes = np.frombuffer(msg['image'], np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame_copy = frame.copy()
        frame, contour_coords = scan_detection(frame)
        warped = four_point_transform(frame_copy, contour_coords)
        _, buffer = cv2.imencode('.jpg', warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        emit('response', {'image': buffer.tobytes()})


if __name__ == '__main__':
    print("Backend running..")
    socketio.run(app, host='localhost', port=5000, debug=True)
