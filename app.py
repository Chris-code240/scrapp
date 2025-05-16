import tkinter
import cv2
import numpy as np
import json
from math import ceil
import customtkinter
from imutils.perspective import four_point_transform
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile, askopenfiles
import utils
from pygrabber.dshow_graph import FilterGraph
import time
import datetime
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

FIELDS = {}
WIDTH = 1266
HEIGHT = 700
app = customtkinter.CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("ScrApp")
app.grid_columnconfigure((0,), weight=1)
app.grid_rowconfigure((0,1,2), weight=1)
back_image = {"image":None, "time":datetime.datetime.now()}
front_image = {"image":None, "time":datetime.datetime.now()}

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
CAMERA_OPTIONS = {**get_cameras(), "Upload Image":-1}
camera_name = customtkinter.StringVar(value=list(CAMERA_OPTIONS.keys())[0])
camera_index = list(CAMERA_OPTIONS.values())[0]
cap = cv2.VideoCapture(list(CAMERA_OPTIONS.values())[0])
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
desired_height = int(original_height * 1)
desired_width = int(WIDTH/3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

_,frame_read = cap.read()
frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
frame_read = cv2.resize(frame_rgb, (desired_width, desired_height))
captured_image = Image.fromarray(frame_read)
window_frame = customtkinter.CTkImage(captured_image,size=(desired_width, desired_height))

def handle_file_upload():
    global capturedImageLabel
    FlashMessage("Select Front Image First!","yellow")
    try:
        files = askopenfiles(filetypes=[("Image Files","*.png *.jpg *.jpeg"),])
        if len(files) == 1:
            FlashMessage("One Image", "yellow")
        elif len(files) != 2 and len(files) != 1:
            FlashMessage("Select Two Images")
            return
        for index, l in enumerate([capturedImageLabel, capturedImageLabel2]):
            image = utils.draw_field_boxes(utils.align_images(image_to_be_aligned_path=files[index].name,reference_iamge_path=f"assets/{'front' if index == 0 else 'back'}.jpg"),index+1)
            warped = cv2.resize(image, (desired_width, desired_height))

            captured_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            temp_window_frame = customtkinter.CTkImage(captured_image, size=(desired_width, desired_height))

            l.configure(image=temp_window_frame, text="")
            l.update()
            FlashMessage(f"{'Front' if index== 0 else 'Back'} Image is Set")
    except Exception as e:
        FlashMessage(str(e),"red")
        capturedImageLabel.configure(text="Error")
        capturedImageLabel.update()
def set_camera_index(choice):
    global camera_index
    global cap
    global cameraSourceComboBox
    global camera_name
    if CAMERA_OPTIONS[choice] != -1:
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
    else:
        handle_file_upload()
def open_camera():
    A4_WIDTH, A4_HEIGHT = 794, 1123
    FACTOR = 0.4
    width, height =ceil( A4_WIDTH * FACTOR),ceil( A4_HEIGHT * FACTOR)
    global window_frame
    global label
    global frame_read
    try:
        if cap is None:
            label.configure(text="Camera Not Initialized")
            label.update()
        else:
            _, frame = cap.read()
            f_height, f_width, ch = frame.shape
            center = (int(f_width/2), int(f_height/2))
            
            # Calculate the top-left and bottom-right corners of the square
            top_left = (center[0] - int(width/2), center[1] - int(height/2))
            bottom_right = (center[0] + int(width/2), center[1] + int(height/2))
            # frame = cv2.filter2D(frame, -1, kernel)
            frame_read = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        if _:
            # Convert the OpenCV frame to PIL Image
            frame = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            captured_image = Image.fromarray(frame)
            window_frame = customtkinter.CTkImage(captured_image, size=(desired_width, desired_height))
            label.configure(image=window_frame)

        label.after(10, open_camera)
    except Exception as e:
        print(e)

# NavFRame
NavFrame = customtkinter.CTkFrame(app)
NavFrame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
NavFrame.grid_columnconfigure((0,1,2), weight=1)
cameraSourceComboBox = customtkinter.CTkComboBox(NavFrame, values=list(CAMERA_OPTIONS.keys()), variable=camera_name,command=set_camera_index)
cameraSourceComboBox.grid(row=0, column=0, sticky="w", padx=10, pady=10)
def set_download_option(choice):
    global download_option
    download_option = choice

# download options
download_option = customtkinter.StringVar(value=OPTIONS[0])
downloadOptionCombo = customtkinter.CTkComboBox(NavFrame, values=OPTIONS,variable=download_option,command=set_download_option )
downloadOptionCombo.grid(row=0, column=2, sticky="e", padx=10, pady=10)

# Flash Message

flashMessage = customtkinter.CTkLabel(NavFrame, width=desired_width, bg_color="transparent",text="", text_color="green")
flashMessage.grid(row=0, column=1,sticky="ew")
def clear_flash_message():
    flashMessage.configure(text="")
    flashMessage.update() 
def FlashMessage(message:str, color="green"):
    flashMessage.configure(text=message, text_color=color)
    flashMessage.update()
    app.after(10000, clear_flash_message)


#Media Frame
MediaFrame = customtkinter.CTkFrame(app)
MediaFrame.grid(row=1, padx=5, pady=5,sticky="ew")
MediaFrame.grid_columnconfigure((0,1,2), weight=1)


# Live feed
label = customtkinter.CTkLabel(MediaFrame,corner_radius=20,image=window_frame, width=desired_width, height=desired_height, bg_color="transparent", text="")
label.grid(row=1, column=0,columnspan=1, sticky="w", pady=10)
open_camera()

# capture button

def captureCallback():
    global capturedImageLabel
    global window_frame
    if  front_image["time"] < back_image["time"] or front_image["image"] is  None:
        capturedImageLabel.configure(image=window_frame, text="")
        capturedImageLabel.update()
        front_image["image"] = cv2.resize(frame_read, (1414, 2000))

        front_image["time"] = datetime.datetime.now()
        FlashMessage("Front Image Set")

    else:
        capturedImageLabel2.configure(image=window_frame, text="")
        capturedImageLabel2.update()
        back_image["image"] = cv2.resize(frame_read, (1414, 2000))
        back_image["time"] = datetime.datetime.now()
        FlashMessage("Back Image Set")

    


# Captured Image
capturedImageLabel = customtkinter.CTkLabel(MediaFrame, text="Front", width=desired_width-10, height=desired_height, bg_color="transparent")
capturedImageLabel.grid(row=1, column=1,columnspan=1, sticky="ew", padx=10, pady=10)
capturedImageLabel2 = customtkinter.CTkLabel(MediaFrame, text="Back", width=desired_width-10, height=desired_height, bg_color="transparent")
capturedImageLabel2.grid(row=1, column=2,columnspan=1, sticky="e", padx=10, pady=10)
#Buttons Frame
ButtonsFrame = customtkinter.CTkFrame(app)
ButtonsFrame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
ButtonsFrame.grid_columnconfigure((0,1), weight=1)

#capturedButton

captureButton = customtkinter.CTkButton(ButtonsFrame, text="Capture", command=captureCallback, width=desired_width)
captureButton.grid(row=0,column=0,padx=10,columnspan=1, pady=10, sticky="w")


#AI Button
def AIButtonCallBack():
    FlashMessage("Not Implemented","yellow")
    pass
GroupedButtonsFrame = customtkinter.CTkFrame(ButtonsFrame)
GroupedButtonsFrame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
GroupedButtonsFrame.grid_columnconfigure((0,1), weight=1)
AIButton = customtkinter.CTkButton(GroupedButtonsFrame, text="Use AI", command=AIButtonCallBack, width=int(desired_width/2))
AIButton.grid(row=0,column=0,padx=10, columnspan=1, pady=10, sticky="w")

def ScrapButtonCallBack():

    FlashMessage("Not Implemented" , "yellow") if front_image["image"] is not None and back_image["image"] is not None else FlashMessage("Front and Back Images Must Be Set","red")

ScrapButton = customtkinter.CTkButton(GroupedButtonsFrame, text="Scrap", command=ScrapButtonCallBack, width=int(desired_width/2), fg_color="green")
ScrapButton.grid(row=0, column=1,columnspan=1, padx=10, pady=10,sticky="e")

app.mainloop()