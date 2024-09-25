import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load data
df = pd.read_csv('reviews_dataset.csv')
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['product_name'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['product_name'])

# Initialize tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['review'], padding="max_length", truncation=True)

# Convert to Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Label encoding
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['product_name'])
val_labels = label_encoder.transform(val_df['product_name'])
test_labels = label_encoder.transform(test_df['product_name'])

train_dataset = train_dataset.add_column("labels", train_labels)
val_dataset = val_dataset.add_column("labels", val_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

# Load model
num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define compute metrics
def compute_metrics(p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate
results = trainer.evaluate(test_dataset)
print(results)

# Save the model and tokenizer
model.save_pretrained("product-retrieval-model")
tokenizer.save_pretrained("product-retrieval-model")
