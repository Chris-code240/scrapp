import tkinter
import cv2
import numpy as np
import json
import customtkinter
from imutils.perspective import four_point_transform
from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

WIDTH = 720
HEIGHT = 480
camera_window_factor = 0.4
app = customtkinter.CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("ScrApp")
app.grid_columnconfigure(0, weight=1) 
app.grid_columnconfigure(1, weight=2)
def get_cameras()->dict:
    devices = FilterGraph().get_input_devices()
    cameras = {}
    for device_index, name in enumerate(devices):
        cameras[name] = device_index
    return cameras
    
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

OPTIONS = ["Download Options","JSON", "Image", "PDF"]
CAMERA_OPTIONS = get_cameras()
camera_name = customtkinter.StringVar(value=list(CAMERA_OPTIONS.keys())[0])
cap = cv2.VideoCapture(list(CAMERA_OPTIONS.values())[0])
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
desired_height = int(original_height * 0.7)
desired_width = int(original_width * (desired_height / original_height))
# desired_width = int(original_width * 0.5)
# desired_width = int(desired_height/original_height)

_,frame_read = cap.read()
frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
frame_read = cv2.resize(frame_rgb, (desired_width, desired_height))
captured_image = Image.fromarray(frame_read)
window_frame = customtkinter.CTkImage(captured_image,size=(desired_width, desired_height))

def set_camera_index(choice):
    global camera_index
    global cap
    global cameraSourceComboBox
    global camera_name
    camera_index = CAMERA_OPTIONS[choice]
    cap.release()
    cap = cv2.VideoCapture(camera_index)
    if not cap.read()[0]:
        for cam in CAMERA_OPTIONS:
            if cv2.VideoCapture(CAMERA_OPTIONS[cam])[0]:
                cap = cv2.VideoCapture(CAMERA_OPTIONS[cam])
                camera_index = CAMERA_OPTIONS[cam]
                camera_name.set(cam)
                cameraSourceComboBox.configure(value=cam)
                cameraSourceComboBox.update()
                break
def open_camera():
    global window_frame
    global label
    if cap is None:
        print("Camera not initialized.")
    else:
        _, frame = cap.read()
        if _:
            # Convert the OpenCV frame to PIL Image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_image = Image.fromarray(frame)
            window_frame = customtkinter.CTkImage(captured_image, size=(desired_width, desired_height))
            label.configure(image=window_frame)

        label.after(10, open_camera)  # Keep updating the camera feed

cameraSourceComboBox = customtkinter.CTkComboBox(app, values=list(CAMERA_OPTIONS.keys()), variable=camera_name,command=set_camera_index)
cameraSourceComboBox.grid(row=0, column=0, sticky="w", padx=10, pady=10)
def set_download_option(choice):
    global download_option
    download_option = choice

# download options
download_option = customtkinter.StringVar(value=OPTIONS[0])
downloadOptionCombo = customtkinter.CTkComboBox(app, values=OPTIONS,variable=download_option,command=set_download_option )
downloadOptionCombo.grid(row=0, column=1, sticky="e", padx=10, pady=10, columnspan=1)

# Live feed
label = customtkinter.CTkLabel(app,corner_radius=20,image=window_frame, bg_color="transparent", text="")
label.grid(row=1, column=0,columnspan=1, sticky="ew", pady=10)
open_camera()

# capture button

def captureCallback():
    global capturedImageLabel
    global window_frame
    capturedImageLabel.configure(image=window_frame, text="")
    capturedImageLabel.update()
captureButton = customtkinter.CTkButton(app, text="Capture", command=captureCallback)
captureButton.grid(row=2,column=0,padx=10, pady=10,columnspan=1, sticky="ew")

# Captured Image
capturedImageLabel = customtkinter.CTkLabel(app, text="Nothing To See", width=desired_width, height=desired_height, bg_color="transparent")
capturedImageLabel.grid(row=1, column=1,columnspan=1, sticky="ew", padx=10, pady=10)
app.mainloop()