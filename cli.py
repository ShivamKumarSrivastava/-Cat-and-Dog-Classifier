import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

def predict_image(img_path,model_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return

    try:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)

        print(f"Processing image {img_path} ...")
        img = Image.open(img_path)
        img = img.resize((128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array,axis = 0)

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            print(f"Prediction: DOG (Confidence: {prediction:.2f})")
        else:
            print(f"Prediction: CAT (Confidence: {1-prediction:.2f})")
    
    except Exception as e:
        print(f"Error during Prediction: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an image contains a dog or cat")
    parser.add_argument('image_path',type=str,help='Path to the image file')
    parser.add_argument('--model',type=str,default='dog_cat_final_model.keras',
    help = 'Path to the model file (default: dog_cat_classification_model.keras)')

    args = parser.parse_args()

    predict_image(args.image_path,args.model)

    