# Heart Disease Prediction API

This project is a heart disease prediction model using TensorFlow, wrapped in a Flask API. The model predicts the likelihood of heart disease based on various medical attributes.

## Project Structure

cvd_prediction_api
├── .venv # Virtual environment directory
├── heart.csv # Dataset file
├── README.md # Project README file
├── train.py # Script to train the model
└── app.py # Flask API script

## Dataset

The dataset (`heart.csv`) contains the following columns:

- age
- sex
- chest pain type (4 values)
- resting blood pressure
- serum cholesterol in mg/dl
- fasting blood sugar > 120 mg/dl
- resting electrocardiographic results (values 0,1,2)
- maximum heart rate achieved
- exercise induced angina
- oldpeak = ST depression induced by exercise relative to rest
- the slope of the peak exercise ST segment
- number of major vessels (0-3) colored by flouroscopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversible defect
- target: values (0,1)

## Setup and Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd cvd_prediction_api
   ```

2. **Create and activate a virtual environment**:
   If virtual environment pacakage is not installed
   ```bash
   pip install virtualenv
   ```
   
   On **Windows**:

   ```bash
   virtualenv .venv
   .venv\Scripts\activate
   ```

   On **macOS/Linux**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. **Install the required packages** from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the model, run the `train.py` script. This script loads the dataset, preprocesses the data, builds and trains a neural network model, and saves the trained model and scaler.

```bash
python train.py
```

## Running the API

To run the Flask API, use the `app.py` script. This script loads the trained model and scaler, and sets up an API endpoint for predictions.

```bash
python app.py
```

### API Endpoint

- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: JSON object with the features array
- **Output**: JSON object with the prediction result

**Sample Input**:

```json
{
  "features": [52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 1, 0, 0]
}
```

**Sample Output**:

```json
{
  "prediction": 1
}
```

## Contributing

Contributions are welcome! If you would like to contribute, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**: `git checkout -b feature-branch-name`
3. **Commit your changes**: `git commit -m 'Add some feature'`
4. **Push to the branch**: `git push origin feature-branch-name`
5. **Create a pull request**.

### Changes Made:
