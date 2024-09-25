
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

3. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the custom dataset:**
   - Ensure that your dataset is saved in CSV format in the `data` directory.
   - The dataset should contain the necessary fields for training the NER model.

## Fine-tuning the BERT model

To fine-tune the BERT model on your custom dataset, run the `train.py` script:

```bash
python train.py
