import asyncio
import websockets
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
    return cv2.drawContours(image_copy, [document_contour], -1, (0, 255, 0), 3),document_contour.reshape(4, 2)


async def process_frame(websocket, path=None):
        async for message in websocket:
            try:
                # Decode JSON message
                data = json.loads(message)

                # Check if "scrap" is in the data
                if data["scrap"]:

                    await asyncio.sleep(0) # Debugging
                    await websocket.send(json.dumps([]))
                else:
                    # Extract image bytes
                    file_bytes = np.frombuffer(np.array(data["image"], dtype=np.uint8).tobytes(), np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    # Process frame
                    frame, contour_coords = scan_detection(frame)
                    warped = four_point_transform(frame, contour_coords)
                    _, buffer = cv2.imencode('.jpg', warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    # Send processed image back
                    await asyncio.sleep(0)
                    await websocket.send(buffer.tobytes())

            except (json.JSONDecodeError, KeyError) as e:
                print("Error decoding message:", e)



        # Send JSON response
            # await websocket.send(cv2.imencode('.jpg', , [int(cv2.IMWRITE_JPEG_QUALITY), 95]))

async def main():
    async with websockets.serve(process_frame, "localhost", 5000, ping_interval=60, ping_timeout=240):
        await asyncio.Future()  # Keeps the server running indefinitely

if __name__ == '__main__':
    print("Websocket is running...")
    asyncio.run(main())
    # print(scan_detection(cv2.imread("../Document_scanner-main\scanned_0.jpg")))
