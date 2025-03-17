import asyncio
import websockets
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import os


WIDTH, HEIGHT = 1080,  1920
def scan_detection(image):
    """
    Detects and draw contours (border-lines) in a frame
    """
    global document_contour
    IMAGE_COPY = image.copy()
                                #[Top-Left],[Top-Right], [Bottom-Left], [Bottom-Right]
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])


    #covert image to gray-scale, then blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #find contours
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
    return cv2.drawContours(IMAGE_COPY, [document_contour], -1, (0, 255, 0), 3)





async def process_frame(websocket, path=None):
    async for message in websocket:
        file_bytes = np.frombuffer(message, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame_copy = frame.copy()
        # Apply some processing (example: grayscale)
        # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scan_detection(frame)
        processed_frame = four_point_transform(frame_copy, document_contour.reshape(4, 2))

        # Convert processed frame back to JPEG
        _, buffer = cv2.imencode('.jpg', processed_frame)
        await websocket.send(buffer.tobytes())

async def main():
    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(process_frame, "", port):
        await asyncio.Future()  # Keeps the server running indefinitely

if __name__ == '__main__':
    print("Websocket is running...")
    asyncio.run(main())
