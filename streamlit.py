import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the trained model and tokenizer
st.title("Product Name Prediction from Review")
st.write("This app predicts the product name based on a product review.")

@st.cache_resource
def load_model_and_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model

# Load the model and tokenizer
model_name_or_path = "product-retrieval-model"
tokenizer, model = load_model_and_tokenizer(model_name_or_path)
model.eval()

@st.cache_resource
def recreate_label_encoder(data_path):
    df = pd.read_csv(data_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['product_name'])
    return label_encoder

# Load dataset to recreate label encoder
label_encoder = recreate_label_encoder('reviews_dataset_1000.csv')

# Function to predict the product name from a review
def predict_product_name(review):
    # Tokenize the input review
    inputs = tokenizer(review, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    # Print the tokenized input for debugging
    print("Tokenized inputs:", inputs)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print model outputs for debugging
    print("Model outputs (logits):", outputs.logits)

    # Get the predicted label (product name)
    predictions = outputs.logits.argmax(-1).item()
    
    # Decode the label to the product name
    product_name = label_encoder.inverse_transform([predictions])[0]
    
    return product_name

# Streamlit interface for user input and prediction
st.write("Enter your product review below:")

review_input = st.text_area("Product Review", value="", height=200)

if st.button("Predict Product Name"):
    if review_input.strip() != "":
        # Run the prediction
        predicted_product = predict_product_name(review_input)
        st.write(f"**Predicted Product Name:** {predicted_product}")
    else:
        st.write("Please enter a valid review.")

# To run the app, use: streamlit run <script_name.py>
