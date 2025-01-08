import keras # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore

def load_and_prep_image(image_path):
    # load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # resize image
    img = cv2.resize(img, (64, 64))
    
    # normalize image
    img = img / 255.0
    
    return np.expand_dims(img, axis=0)

def predict_sign(model_path, image_path):
    # Load model using keras instead of tf
    model = keras.models.load_model(model_path)
    
    # Prepare image
    img = load_and_prep_image(image_path)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Get class names
    class_names = ['Stop Sign', 'Traffic Light', 'Other']
    
    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

if __name__ == "__main__":
    model_path = "model/traffic_sign_model.h5"
    # use a specific image from test directory
    image_path = "data/test/stop_signs/00014_00000_00000.png"  # this file actually exists in directory
    
    predicted_class, confidence = predict_sign(model_path, image_path)
    print(f"Image path: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
