# IDR Currency Classifier (Streamlit + TensorFlow)

This is a machine learning web app built with Streamlit and TensorFlow to classify Indonesian Rupiah banknotes from uploaded images.

## Live App

Visit the app: [idr-currency-classification.streamlit.app](https://idr-currency-classification.streamlit.app/)

Note: If the app doesn't load immediately, it might be inactive. Use the "Reboot" button on Streamlit.

## Features

- Image upload for currency classification.  
- Pretrained MobileNetV2-based model for Indonesian Rupiah denominations.
- Shows top prediction with confidence score.
- Displays top 5 class probabilities in an expandable section.
- Environment-based configuration with `.env` for model path.
- Efficient model loading with caching.

## Model Details

The model was trained on the **[2022 Indonesian Rupiah banknotes dataset](https://www.kaggle.com/datasets/fannyzahrahramadhan/uang-emisi-2022-baru)** which is available on Kaggle. This ensures accurate recognition of the most recent banknote designs.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tengkumuazabs/idr-currency-classification.git
cd idr-currency-classifier
```

### 2. Create `.env` File
```bash
echo "MODEL_PATH=mobilenetv2_custom_model.keras" > .env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app locally
```bash
streamlit run app.py
```

## Inputs and Output
- Upload an image of an Indonesian Rupiah banknote (png, jpg, jpeg).
- The app preprocesses the image and runs it through the model.
- Displays the predicted banknote denomination with confidence score.
- Shows the top 5 predictions and their probabilities.

