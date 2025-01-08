import keras # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
import matplotlib.pyplot as plt # type: ignore

# match constants with train.py
IMG_HEIGHT = 64
IMG_WIDTH = 64

def load_and_prep_image(image_path):
    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # resize image to match training size
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    
    # normalize image
    img = img / 255.0
    
    return np.expand_dims(img, axis=0)

def predict_sign(model_path, image_path):
    try:
        # load model
        model = keras.models.load_model(model_path)
        
        # prepare image
        img = load_and_prep_image(image_path)
        
        # make prediction
        prediction = model.predict(img, verbose=0)
        
        # IMPORTANT: verify if this order matches your training data exactly
        class_names = ['other', 'stop_signs', 'traffic_lights']  # changed order to test
        
        # add debugging information
        print("\nDebug Information:")
        print("Raw prediction values:", prediction[0])
        for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
            print(f"{class_name}: {prob*100:.2f}%")
        
        # get predicted class and confidence
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx] * 100)
        
        # convert class names to display format
        if predicted_class == 'stop_signs':
            predicted_class = 'Stop Sign'
        elif predicted_class == 'traffic_lights':
            predicted_class = 'Traffic Light'
        elif predicted_class == 'other':
            predicted_class = 'Other Sign'
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

if __name__ == "__main__":
    model_path = "model/traffic_sign_model.h5"
    image_path = "data/test/stop_signs/00014_00000_00000.png"
    
    predicted_class, confidence = predict_sign(model_path, image_path)
    print(f"Image path: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
