import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# set page title
st.title("Dog Vs Cat Classifier App")

# load the pre-trained model
@st.cache_resource
def load_classification_model():
    model = load_model('dog_cat_final_model.keras')
    return model

try:
    model = load_classification_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# file uploader

uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', width=300)

    # preprocess the image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    if st.button("Classify Image"):
        prediction = model.predict(img_array)[0][0]

        # dispaly the result
        if prediction > 0.5:
            st.success(f"This is a Dog! (confidence: {prediction:.2f})")
        else:
            st.success(f"This is a Cat! (confidence: {1 - prediction:.2f})")
        
        # Display prediction probability bar
        st.progress(float(prediction) if prediction > 0.5 else float(1 - prediction))

        # Display Probability Distribution
        st.bar_chart({
            'Cat': [1 - prediction],
            'Dog': [prediction]
        })

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a cat or dog in JPG, JPEG, or PNG format.
2. Click the "Classify Image" button to see the prediction.
3. The app will display whether the image is of a cat or a dog along with the confidence level.
""")

# About
st.sidebar.header("About")
st.sidebar.write("""
This app uses a pre-trained deep learning model to classify images of cats and dogs.    
The model was trained on a dataset of labeled images and can predict using TensorFlow/Keras.
""")
        