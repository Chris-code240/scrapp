import os
from math import ceil
import json
import random
import string
import cv2
import numpy as np
import shutil
from imutils.perspective import four_point_transform
from spire.pdf.common import *
from spire.pdf import PdfDocument

def pdf_to_jpg():
    # Create a PdfDocument object
    pdf = PdfDocument()

    # Load a PDF document
    pdf.LoadFromFile("tesseract_train.pdf")

    # Iterate through all pages in the document
    for i in range(pdf.Pages.Count):

        # Save each page as an PNG image
        fileName = "assets\\ToImage-{0:d}.png".format(i)
        with pdf.SaveAsImage(i) as imageS:
            imageS.Save(fileName)
    pdf.Close()
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


fields = []

def draw_field_boxes(image,page=1):
    with open(f"assets/{'front' if page == 1 else 'back'}.json", 'r') as file:
        fields = json.load(file)

    for f in fields[0]['annotations']:
        f = f['coordinates']
        x_min = ceil(f['x'] - (f['width']/ 2))
        y_min = ceil(f['y'] - (f['height']/2))
        x_max = ceil(f['x']+ f['width'] - (f['width']/ 2))
        y_max = ceil(f['y']+f['height'] - (f['height']/2))
        image = cv2.rectangle(image, (x_min ,y_min),(x_max, y_max),(10, 200, 5), 2)
        # image = cv2.rectangle(img=image, pt1=(ceil(f['x']), ceil(f['y'])), pt2=(ceil(f['x']) + ceil(f['width']), ceil(f['y']) + ceil(f['height'])),thickness=2,lineType=-1, color=(10, 200, 5))
    return image
def generate_random_string(length=12):
    characters = string.ascii_letters + string.digits  # Includes both letters and digits
    return ''.join(random.choice(characters) for _ in range(length))
def crop_fields(image):
    images = []
    for f in fields[0]['annotations']:
        f = f['coordinates']
        x_min = ceil(f['x'] - (f['width']/ 2))
        y_min = ceil(f['y'] - (f['height']/2))
        x_max = ceil(f['x']+ f['width'] - (f['width']/ 2))
        y_max = ceil(f['y']+f['height'] - (f['height']/2))
        images.append(image[y_min:y_max, x_min:x_max]) 
    print(len(images)) 
    return images if len(images) > 1 else []

def json_to_yolo_format(p="dataset_fullpage"):
        # Define paths
    dataset_dir = p
    DIR = dataset_dir
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    parent_dir = dataset_dir  # JSON files and original images are here
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    splits = ["train", "val"]

    # Ensure labels directories exist
    split_factor = 0.9
    images = [i for i in os.listdir(dataset_dir) if i.endswith(('.jpg', '.jpeg', '.png'))]
    train_images = random.sample(images, int(split_factor * len(images)))
    val_images = [i for i in images if i not in train_images]
    for split in splits:
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Function to convert bounding box to YOLO format
    def convert_to_yolo_format(x_center, y_center, width, height, img_width, img_height):
        # Normalize by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        return x_center_norm, y_center_norm, width_norm, height_norm

    # Process each split (train and val)
    for split in splits:
        # Process each image in the split
        for img_file in os.listdir(dataset_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Look for the JSON file in the parent directory
            json_file = img_file[:len(img_file)-4] + ".json"
            json_path = os.path.join(dataset_dir, json_file)
            
            if not os.path.exists(json_path):
                print(f"JSON file not found for {img_file} at {json_path}, skipping...")
                continue
            
            # Load the image from images/train/ or images/val/ to get its dimensions
            img_path = os.path.join(dataset_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}, skipping...")
                continue
            img_height, img_width = img.shape[:2]
            
            # Read the JSON file
            with open(json_path, "r") as f:
                data = json.load(f)
            
            # Since the JSON is a list with one dictionary, get the first item
            if not data or len(data) == 0:
                print(f"Empty JSON file for {img_file}, skipping...")

                _in = input("Continue? ")
                if _in.lower() == "y" or "yes":
                    continue
                else:
                    break
            
            image_data = data[0]
            
            # Verify the image name matches
            if image_data["image"] != img_file and image_data["image"] != img_file[2:]:
                print(f"JSON image name {image_data['image']} does not match file {img_file}, skipping...")
                _in = input("Continue? ")
                if _in.lower() == "y" or "yes":
                    continue
                else:
                    break
            
            # Process each annotation
            yolo_annotations = []
            for annotation in image_data["annotations"]:
                label = annotation["label"]
                coords = annotation["coordinates"]
                
                # Extract coordinates (center-based)
                x_center = coords["x"]
                y_center = coords["y"]
                width = coords["width"]
                height = coords["height"]
                
                # Determine the class ID based on the label
                if label.endswith("__name") or 'n' in label.split("__")[-1]:
                    class_id = 0  # field_name
                elif label.endswith(("__box", "__textbox","_textbox")) or 't' in label.split("__")[-1] or  'b' in label.split("__")[-1]:
                    class_id = 1  # handwritten_text
                else:
                    print(f"Unknown label format: {label} in {img_file}, skipping...")
                    continue
                
                # Convert to YOLO format (normalize coordinates)
                x_center_norm, y_center_norm, width_norm, height_norm = convert_to_yolo_format(
                    x_center, y_center, width, height, img_width, img_height
                )
                
                # Add to annotations
                yolo_annotations.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
            
            # Save the YOLO annotations to a .txt file in labels/train/ or labels/val/
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(DIR, txt_file)
            with open(txt_path, "w") as f:
                f.writelines(yolo_annotations)
            if img_file in train_images:
                shutil.copy(img_path,os.path.join(dataset_dir,"images/train"))
                shutil.copy(txt_path,os.path.join(dataset_dir,"labels/train"))
            else:
                shutil.copy(img_path,os.path.join(dataset_dir,"images/val"))
                shutil.copy(txt_path,os.path.join(dataset_dir,"labels/val"))  
            print(f"Saved YOLO annotations to {txt_path}")



def distinguish_files(p="ldataset_fullpage - Copy"):
    n = random.randint(128, 256)
    for d in os.listdir(p):
        if os.path.isfile(f"{p}/{d}"):
            new = f"{n}_{d}"
            data = None
            if d.endswith(".json"):
                with open(f"{p}/{d}", "r") as file:
                    data = json.load(file)
                data[0]["image"] = new
                with open(f"{p}/{d}", "w") as file:
                    json.dump(data,file, indent=4)
            os.rename(f"{p}/{d}", f"{p}/{new}")
    print(n)

def remove_json_extension(p="ldataset_fullpage - Copy"):
    for d in os.listdir(p):
        if os.path.isfile(f"{p}/{d}"):
            data = None
            if d.endswith(".json"):
                with open(f"{p}/{d}", "r") as file:
                    data = json.load(file)
                data[0]["image"] = data[0]["image"].replace("..", ".")
                with open(f"{p}/{d}", "w") as file:
                    json.dump(data,file, indent=4)

def check_annotations():


    # Paths
    split = "train"  # Change to "val" to check validation images
    images_dir = f"dataset_fullpage/"
    labels_dir = f"dataset_fullpage/"
    output_dir = f"dataset_fullpage/annotated_{split}/"
    os.makedirs(output_dir, exist_ok=True)

    # Class names
    class_names = ["field_name", "handwritten_text"]

    # Process each image
    for img_file in os.listdir(images_dir):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Load the image
        img_path = os.path.join(images_dir, img_file)

        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        # Load the annotations
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_file)
        if not os.path.exists(txt_path):
            print(f"Annotation file not found for {img_file}, skipping...")
            continue
        
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()
        
            # Draw bounding boxes
            print("Len: ", len(lines))
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
                class_id = int(class_id)
                
                # Convert from normalized to absolute coordinates
                x_center = x_center * img_width
                y_center = y_center * img_height
                width = width * img_width
                height = height * img_height
                
                # Calculate top-left and bottom-right corners
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                # Draw the bounding box
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for field_name, Red for handwritten_text
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                img = cv2.putText(img, class_names[class_id], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(e, "File: ", img_file)
        # Save the annotated image
        output_path = os.path.join(output_dir, img_file)
        cv2.imshow("Frame", img)
        cv2.waitKey(0)

def check_class_imbalance():
    # Count class distribution in training annotations
    labels_dir = "dataset_fullpages/labels/train"
    class_counts = {0: 0, 1: 0}

    for txt_file in os.listdir(labels_dir):
        if not txt_file.endswith(".txt"):
            continue
        txt_path = os.path.join(labels_dir, txt_file)
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            class_id = int(line.strip().split()[0])
            class_counts[class_id] += 1

    print(f"Number of field_name (class 0): {class_counts[0]}")
    print(f"Number of handwritten_text (class 1): {class_counts[1]}")


def move_from_canvas_to_dir(destination_dir:str="dataset_fullpage/", canvas_dir="from_canvas/"):
    files_read = []
    destination_dir = destination_dir.replace('\\','/')
    destination_dir = destination_dir + '/' if destination_dir[-1] != '/' else destination_dir
    os.makedirs(destination_dir,exist_ok=True)
    canvas_files = os.listdir(canvas_dir)

    for d in canvas_files:
        if os.path.isfile(d):
            if not d.endswith('.json') and os.path.exists(canvas_dir+d[:len(d)-4] + '.jpg') and d[:len(d)-4] not in files_read:
                shutil.copyfile(canvas_dir +d[:len(d)-4] + '.jpg', destination_dir+d[:len(d)-4] + '.jpg')
                shutil.copyfile(canvas_dir +d[:len(d)-4] + '.json', destination_dir+d[:len(d)-4] + '.json')
                files_read.append(d[:len(d)-4])


def check_annotation_predicted(txt_path="", img_path=""):
        path = txt_path[:len(txt_path)-4]
        img = cv2.imread(img_path)
        img_height, img_width, chn = img.shape
        class_names = {0:"field name", 1:"handwritten"}

        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()
                print(lines)
        
            # Draw bounding boxes
            print("Len: ", len(lines))
            for line in lines:
                class_id, x_center, y_center, width, height, conf = map(float, line.strip().split())
            
                class_id = int(class_id)
                
                
                # Convert from normalized to absolute coordinates
                x_center = x_center * img_width
                y_center = y_center * img_height
                width = width * img_width
                height = height * img_height
                
                # Calculate top-left and bottom-right corners
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                # Draw the bounding box
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for field_name, Red for handwritten_text
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                img = cv2.putText(img, class_names[class_id] +" "+ str(conf), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(e, "File: ")
        # Save the annotated image
        cv2.imwrite("Predicted_5_back.jpg",img)
        desired_height = int(img_height * 0.7)
        desired_width = int(img_width*0.4)
        cv2.imshow("Frame", cv2.resize(img, (desired_width, desired_height)))
        cv2.waitKey(0)


def json_to_yolo_format(p="dataset_fullpage"):
        # Define paths
    dataset_dir = p
    DIR = dataset_dir
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    parent_dir = dataset_dir  # JSON files and original images are here
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    splits = ["train", "val"]

    # Ensure labels directories exist
    split_factor = 0.9
    images = [i for i in os.listdir(dataset_dir) if i.endswith(('.jpg', '.jpeg', '.png'))]
    train_images = random.sample(images, int(split_factor * len(images)))
    val_images = [i for i in images if i not in train_images]
    for split in splits:
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Function to convert bounding box to YOLO format
    def convert_to_yolo_format(x_center, y_center, width, height, img_width, img_height):
        # Normalize by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        return x_center_norm, y_center_norm, width_norm, height_norm

    # Process each split (train and val)
    for split in splits:
        # Process each image in the split
        for img_file in os.listdir(dataset_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Look for the JSON file in the parent directory
            json_file = img_file[:len(img_file)-4] + ".json"
            json_path = os.path.join(dataset_dir, json_file)
            
            if not os.path.exists(json_path):
                print(f"JSON file not found for {img_file} at {json_path}, skipping...")
                continue
            
            # Load the image from images/train/ or images/val/ to get its dimensions
            img_path = os.path.join(dataset_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}, skipping...")
                continue
            img_height, img_width = img.shape[:2]
            
            # Read the JSON file
            with open(json_path, "r") as f:
                data = json.load(f)
            
            # Since the JSON is a list with one dictionary, get the first item
            if not data or len(data) == 0:
                print(f"Empty JSON file for {img_file}, skipping...")

                _in = input("Continue? ")
                if _in.lower() == "y" or "yes":
                    continue
                else:
                    break
            
            image_data = data[0]
            
            # Verify the image name matches
            if image_data["image"] in img_file and image_data["image"] != img_file[2:]:
                print(f"JSON image name {image_data['image']} does not match file {img_file}, skipping...")
                _in = input("Continue? ")
                if _in.lower() == "y" or "yes":
                    continue
                else:
                    break
            
            # Process each annotation
            yolo_annotations = []
            for annotation in image_data["annotations"]:
                label = annotation["label"]
                coords = annotation["coordinates"]
                
                # Extract coordinates (center-based)
                x_center = coords["x"]
                y_center = coords["y"]
                width = coords["width"]
                height = coords["height"]
                
                # Determine the class ID based on the label
                if label.endswith("__name") or 'n' in label.split("__")[-1]:
                    class_id = 0  # field_name
                elif label.endswith(("__box", "__textbox","_textbox")) or 't' in label.split("__")[-1] or  'b' in label.split("__")[-1]:
                    class_id = 1  # handwritten_text
                else:
                    print(f"Unknown label format: {label} in {img_file}, skipping...")
                    continue
                
                # Convert to YOLO format (normalize coordinates)
                x_center_norm, y_center_norm, width_norm, height_norm = convert_to_yolo_format(
                    x_center, y_center, width, height, img_width, img_height
                )
                
                # Add to annotations
                yolo_annotations.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
            
            # Save the YOLO annotations to a .txt file in labels/train/ or labels/val/
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(DIR, txt_file)
            with open(txt_path, "w") as f:
                f.writelines(yolo_annotations)
            if img_file in train_images:
                shutil.copy(img_path,os.path.join(dataset_dir,"images/train"))
                shutil.copy(txt_path,os.path.join(dataset_dir,"labels/train"))
            else:
                shutil.copy(img_path,os.path.join(dataset_dir,"images/val"))
                shutil.copy(txt_path,os.path.join(dataset_dir,"labels/val"))  
            print(f"Saved YOLO annotations to {txt_path}")

def distinguish_files(p="ldataset_fullpage - Copy"):
    n = random.randint(12, 128)
    for d in os.listdir(p):
        if os.path.isfile(f"{p}/{d}"):
            new = f"{n}_{d}"
            data = None
            if d.endswith(".json"):
                with open(f"{p}/{d}", "r") as file:
                    data = json.load(file)
                data[0]["image"] = new
                with open(f"{p}/{d}", "w") as file:
                    json.dump(data,file, indent=4)
            os.rename(f"{p}/{d}", f"{p}/{new}")
    print(n)

def remove_json_extension(p="ldataset_fullpage - Copy"):
    for d in os.listdir(p):
        if os.path.isfile(f"{p}/{d}"):
            data = None
            if d.endswith(".json"):
                with open(f"{p}/{d}", "r") as file:
                    data = json.load(file)
                data[0]["image"] = data[0]["image"].replace("..", ".")
                with open(f"{p}/{d}", "w") as file:
                    json.dump(data,file, indent=4)

def check_annotations():


    # Paths
    split = "train"  # Change to "val" to check validation images
    images_dir = f"dataset_fullpage/"
    labels_dir = f"dataset_fullpage/"
    output_dir = f"dataset_fullpage/annotated_{split}/"
    os.makedirs(output_dir, exist_ok=True)

    # Class names
    class_names = ["field_name", "handwritten_text"]

    # Process each image
    for img_file in os.listdir(images_dir):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Load the image
        img_path = os.path.join(images_dir, img_file)

        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        # Load the annotations
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_file)
        if not os.path.exists(txt_path):
            print(f"Annotation file not found for {img_file}, skipping...")
            continue
        
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()
        
            # Draw bounding boxes
            print("Len: ", len(lines))
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
                class_id = int(class_id)
                
                # Convert from normalized to absolute coordinates
                x_center = x_center * img_width
                y_center = y_center * img_height
                width = width * img_width
                height = height * img_height
                
                # Calculate top-left and bottom-right corners
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                # Draw the bounding box
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for field_name, Red for handwritten_text
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                img = cv2.putText(img, class_names[class_id], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(e, "File: ", img_file)
        # Save the annotated image
        output_path = os.path.join(output_dir, img_file)
        cv2.imshow("Frame", img)
        cv2.waitKey(0)

def check_class_imbalance():
    # Count class distribution in training annotations
    labels_dir = "dataset_fullpages/labels/train"
    class_counts = {0: 0, 1: 0}

    for txt_file in os.listdir(labels_dir):
        if not txt_file.endswith(".txt"):
            continue
        txt_path = os.path.join(labels_dir, txt_file)
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            class_id = int(line.strip().split()[0])
            class_counts[class_id] += 1

    print(f"Number of field_name (class 0): {class_counts[0]}")
    print(f"Number of handwritten_text (class 1): {class_counts[1]}")


def move_from_canvas_to_dir(destination_dir:str="dataset_fullpage/", canvas_dir="from_canvas/"):
    files_read = []
    destination_dir = destination_dir.replace('\\','/')
    destination_dir = destination_dir + '/' if destination_dir[-1] != '/' else destination_dir
    os.makedirs(destination_dir,exist_ok=True)
    canvas_files = os.listdir(canvas_dir)

    for d in canvas_files:
        if os.path.isfile(d):
            if not d.endswith('.json') and os.path.exists(canvas_dir+d[:len(d)-4] + '.jpg') and d[:len(d)-4] not in files_read:
                shutil.copyfile(canvas_dir +d[:len(d)-4] + '.jpg', destination_dir+d[:len(d)-4] + '.jpg')
                shutil.copyfile(canvas_dir +d[:len(d)-4] + '.json', destination_dir+d[:len(d)-4] + '.json')
                files_read.append(d[:len(d)-4])


def check_annotation_predicted(txt_path="", img_path=""):
        path = txt_path[:len(txt_path)-4]
        img = cv2.imread(img_path)
        img_height, img_width, chn = img.shape
        class_names = {0:"field name", 1:"handwritten"}

        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()
                print(lines)
        
            # Draw bounding boxes
            print("Len: ", len(lines))
            for line in lines:
                class_id, x_center, y_center, width, height, conf = map(float, line.strip().split())
            
                class_id = int(class_id)
                
                
                # Convert from normalized to absolute coordinates
                x_center = x_center * img_width
                y_center = y_center * img_height
                width = width * img_width
                height = height * img_height
                
                # Calculate top-left and bottom-right corners
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                # Draw the bounding box
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for field_name, Red for handwritten_text
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                img = cv2.putText(img, class_names[class_id] +" "+ str(conf), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(e, "File: ")
        # Save the annotated image
        cv2.imwrite("Predicted_5_back.jpg",img)
        desired_height = int(img_height * 0.7)
        desired_width = int(img_width*0.4)
        cv2.imshow("Frame", cv2.resize(img, (desired_width, desired_height)))
        cv2.waitKey(0)


import numpy as np
def align_images(image_to_be_aligned_path:str, reference_iamge_path:str="assets/front.jpg"):
    # Open the image files.
    img1_color = cv2.imread(image_to_be_aligned_path)  # Image to be aligned.
    img2_color = cv2.imread(reference_iamge_path)    # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.ones((no_of_matches, 2))
    p2 = np.ones((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))

    # Save the output.
    return transformed_img
def reverse_yolo_annotation(txt_file_path:str):
    lines = []
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    
    for index, line in enumerate(lines):
        line = line.strip().split(' ')

        line[0] = str(index)
        lines[index] = ' '.join(line + ['\n'])
    with open(txt_file_path, 'w') as file:
        file.writelines(lines)

def generate_gt_txt(path:str):

    for dir in os.listdir(path):
        num = random.randint(2, 200)
        if dir.endswith(('.jpg', '.png')):
            text = dir.split('__')[0] if '_' in dir else dir.split('.png')[0]
            with open(os.path.join(path, text.replace(' ','_')+f'_{num}.gt.txt'), 'w') as file:
                file.write(text)
            os.rename(os.path.join(path, dir), os.path.join(path, text.replace(' ','_')+f'_{num}.png'))

# json_to_yolo_format("temp")
# distinguish_files("temp")

# cv2.imshow("Frame", draw_field_boxes(align_images("dataset_new_/81_5.jpg", "assets/front.jpg")))
# cv2.waitKey(0)