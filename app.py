import streamlit as st
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import os

# load the model
load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH', 'mobilenetv2_custom_model.keras')

# load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# defining class names (same order with the training)
CLASS_NAMES = ['100RB', '10RB', '1RB', '20RB', '2RB', '50RB', '5RB']

# image preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0     # rescale the image
    if image.shape[-1] == 4:
        image = image[..., :3]          # handle png with the alpha channel
    image = np.expand_dims(image, axis=0)
    return image

col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    st.title('💵IDR Currency Classifier')

    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded image', use_container_width=True)
    
        with st.container():
            input_arr = preprocess_image(img)
            pred = model.predict(input_arr)
            prob = pred[0]
            top_indexes = prob.argsort()[-5:][::-1]      # showing top 5 probabilities

            # final prediction
            top_prediction_index = top_indexes[0]
            top_label = CLASS_NAMES[top_prediction_index]
            top_confidence = prob[top_prediction_index] * 100

            st.markdown('## Final Prediction: ')
            st.success(f'**{top_label}** with **{top_confidence:.2f}%** confidence.')

            with st.expander('Top 5 Prediciton Probabilities', expanded=True):
                top_5_data = {
                    'Class': [CLASS_NAMES[i] for i in top_indexes],
                    'Confidence (%)': [f'{prob[i] * 100:.2f}%' for i in top_indexes]
                }
                st.write('These are the top 5 most likely classes for the uploaded image: ')
                st.dataframe(top_5_data, use_container_width=True)