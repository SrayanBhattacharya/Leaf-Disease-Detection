import streamlit as st
import tensorflow as tf
import numpy as np

def modelPrediction(testImage):
    model = tf.keras.models.load_model("trainedModel.keras")
    image = tf.keras.preprocessing.image.load_img(testImage, target_size=(128, 128))
    inputArr = tf.keras.preprocessing.image.img_to_array(image)
    inputArr = np.array([inputArr]) #Converts single image to batch
    predictions = model.predict(inputArr)
    resultIndex = np.argmax(predictions)
    return resultIndex

#Sidebar
st.sidebar.title("Dashboard")
appMode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Home Page
if appMode == "Home":
    st.header("DEEP LEARNING BASED LEAF DISEASE DETECTION SYSTEM")
    imagePath = "HomeScreen.jpg"
    st.image(imagePath, use_column_width=True)
    st.markdown("""
                Welcome to the Leaf Disease Detection System.

                The main purpose of this system is to detect the disease of a leaf 
                efficiently using machine learning.

                ### How it Works?

                1. **Upload Image:** Upload the image of the leaf to our system.
                2. **Processing:** Our system will process the image using advanced algorithms.
                3. **Results:** After the necessary processing is done the results will be shown and further
                actions will be taken accordingly.

                """)
    
#About Page
elif appMode == "About":
    st.header("ABOUT US")
    st.markdown("""
                ### Dataset
                This dataset is created using offline augmentation from the original dataset. The original
                dataset can be found on: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

                This project was made using the Convolution Neural Network(CNN) model of machine learning. 
""")
    
#Disease Detection page
elif appMode == "Disease Recognition":
    st.header("Disease Recognition")
    testImage = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(testImage, use_column_width=True)
    #Predict Button
    if st.button("Predict"):
        with st.spinner("Please Wait....."):
            st.write("Our Prediction")
            resultIndex = modelPrediction(testImage)
            #Define Class
            className = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
            st.success(f"Model is predicting it's a {className[resultIndex]}")