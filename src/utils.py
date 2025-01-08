import cv2 # type: ignore
import numpy as np # type: ignore
from keras.utils import img_to_array # type: ignore

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    load and preprocess a single image
    """
    # read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # resize
    img = cv2.resize(img, target_size)
    
    # convert to array and normalize
    img = img_to_array(img)
    img = img.astype('float32') / 255.0
    
    return img

def visualize_prediction(image_path, prediction, confidence):
    """
    display image with prediction and confidence
    """
    # read original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # add text to image
    text = f"{prediction}: {confidence:.2f}%"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    return img
