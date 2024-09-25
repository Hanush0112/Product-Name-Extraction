import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model and tokenizer
print("Loading model and tokenizer...")
model_name_or_path = "product-retrieval-model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
print("Model and tokenizer loaded.")

# Load the label encoder (assuming it was saved earlier or you recreate it with the same classes)
print("Loading dataset to recreate label encoder...")
df = pd.read_csv('reviews_dataset_1000.csv')  # Load your dataset to recreate the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(df['product_name'])
print("Label encoder recreated.")

# Set the model to evaluation mode
model.eval()

# Function to predict the product name from a review
def predict_product_name(review):
    print("Tokenizing review...")
    # Tokenize the input review
    inputs = tokenizer(review, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    print("Running model inference...")
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label (product name)
    predictions = outputs.logits.argmax(-1).item()
    
    print("Decoding predicted label...")
    # Decode the label to the product name
    product_name = label_encoder.inverse_transform([predictions])[0]
    
    return product_name

# Function to interactively get user input and predict the product name
def get_user_input():
    review = input("Please enter your product review: ")
    predicted_product = predict_product_name(review)
    print(f"The predicted product name is: {predicted_product}")
while True:
# Example interactive usage
    get_user_input()
    choice = input("do you want to continue(y/n):")
    if choice == 'n':
        break
