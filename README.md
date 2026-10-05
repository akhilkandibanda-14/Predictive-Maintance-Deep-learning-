# Predictive Maintenance of Machines Using Deep Learning

The objective of this project is to predict whether an industrial machine is likely to fail based on sensor and operational data.

## Dataset
We are using the AI4I 2020 Predictive Maintenance Dataset.

## Project Structure
- `dataset/`: Contains the actual machine data
- `notebooks/`: Jupyter notebooks for data exploration and modeling
- `src/`: Source code for the project
- `models/`: Trained machine learning models
- `results/`: Output from models, such as metrics or plots
- `app.py`: Streamlit application for inference
- `requirements.txt`: Python dependencies

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akhilkandibanda-14/Predictive-Maintance-Deep-learning-.git
   cd Predictive-Maintance-Deep-learning-
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Data Preprocessing & Modeling
If you want to re-run the preprocessing and model training from scratch, execute the scripts from the root directory:
```bash
# Run Exploratory Data Analysis
python src/generate_eda.py

# Run Preprocessing
python src/preprocessing.py

# Train LSTM & Compare against ANN
python src/train_lstm.py
```

### 2. Run Inference via Command Line
To test the pre-trained ANN model with sample machine inputs (Healthy vs High Risk) via the terminal:
```bash
python src/predict.py
```

### 3. Run the Streamlit Web Application
To start the interactive web interface:
```bash
streamlit run app.py
```
