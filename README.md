
## Installation

To get started with the project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ner-bert-project.git
    cd ner-bert-project
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```



3. **Prepare the custom dataset:**
   - Ensure that your dataset is saved in CSV format in the `data` directory.
   - The dataset should contain the necessary fields for training the model.

## Fine-tuning the BERT model

To fine-tune the BERT model on your custom dataset, run the `app.py` script:
```bash
python train.py
```

4. **Predicting the product names from the product reviews**
   predict the product names from the product reviews using the created model just run the python file 'prediction.py'

```bash
python prediction.py
```

5. **To create a webpage using a streamlit application**
   ```bash
   streamlit run streamlit.py
   ```
