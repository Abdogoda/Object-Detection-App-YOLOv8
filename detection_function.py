# ------------------------------
# Import Libraries
# ------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# --- DEFINE THE MODEL ---
model = YOLO("yolov8m.pt")

# --- FUNCTION TO DETECT ---
def detection(frame):
    if frame is None: 
        say_text("Frame Not Found")
        return 0
    
    results = model.predict(frame)
    object_counts = get_objects_count(model, results)

    return object_counts


# --- OBJECTS COUNT ---
def get_objects_count(model, results):
 unique_objects = [] # list of unique objects
 object_counts = {} # list of object and its count

 for result in results:
  if result.boxes:
   for box in result.boxes:
    ClassInd = int(box.cls)
    if model.names[ClassInd] not in object_counts:
     unique_objects.append(model.names[ClassInd])
     object_counts[model.names[ClassInd]] = 1
    else:
     object_counts[model.names[ClassInd]] += 1

 return object_counts

# --- Create Text From Object Names ---
def objectNames(object_names_count):
    final_text = ""
    for index, (key, value) in enumerate(object_names_count.items()):
        is_sum = "s" if value > 1 else ""
        is_and = "" if index == len(object_names_count)-1 else "and "
        final_text += f"{value} {key}{is_sum} {is_and}"
 
    return final_text
    


# --- IMAGE DETECTION ---
def image_detection(frame):
    object_counts = detection(frame)
    return objectNames(object_counts)
