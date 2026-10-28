# 🌱 CropSense — Smart Crop Recommendation System

CropSense is a machine learning web application that recommends a crop based on soil nutrient values and environmental conditions. Users enter seven measurements, and the trained classification model predicts a suitable crop category.

The application is built with Python and Streamlit and is intended for educational and agricultural decision-support purposes.

## ✨ Features

- Predicts a crop category from seven input parameters.
- Accepts soil nutrient measurements: Nitrogen (N), Phosphorus (P), and Potassium (K).
- Accepts environmental measurements: temperature, humidity, soil pH, and rainfall.
- Displays the predicted crop in a result card.
- Displays the model's top five class scores when the saved model supports `predict_proba`.
- Shows a summary of the submitted input values.
- Includes a Streamlit web interface with a responsive, green-themed design.
- Loads a previously trained model; model training is performed separately in the notebook.

## 🧪 Input Features

| Feature | Description |
|---|---|
| `N` | Nitrogen value |
| `P` | Phosphorus value |
| `K` | Potassium value |
| `temperature` | Temperature in °C |
| `humidity` | Humidity in % |
| `ph` | Soil pH |
| `rainfall` | Rainfall in mm |

Enter values using the same units and conventions as the dataset used to train the model.

## 🛠️ Technology Stack

- **Language:** Python
- **Web application:** Streamlit
- **Data handling:** pandas
- **Model loading:** joblib
- **Machine learning:** scikit-learn
- **Development environment:** Jupyter Notebook / Visual Studio Code

## 📁 Project Structure

Keep the application and saved model files together in the same project folder:

```text
MI Project/
├── app.py
├── main.ipynb
├── Crop_recommendation.csv
├── best_crop_model.pkl
├── crop_features.pkl
├── crop_model_name.pkl
├── crop_scaler.pkl          # Required only for Logistic Regression or SVM
├── requirements.txt
└── README.md
```

The scaler file is needed only if the selected model is **Logistic Regression** or **SVM**. The app expects the other three `.pkl` files to be present.

> **Important:** The saved `.pkl` model files are required to run the app. Do not upload sensitive or untrusted pickle files. Only load model files that you created or otherwise trust.

## 🚀 Run Locally

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username and `YOUR-REPOSITORY` with your repository name.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the required model files

Make sure these files are beside `app.py`:

- `best_crop_model.pkl`
- `crop_features.pkl`
- `crop_model_name.pkl`

If your selected model is Logistic Regression or SVM, also include:

- `crop_scaler.pkl`

### 5. Start the application

```bash
python -m streamlit run app.py
```

Open the local address printed in the terminal, usually:

```text
http://localhost:8501
```

## 🧠 Model and Prediction Workflow

1. Load and inspect the crop recommendation dataset.
2. Check for missing values and duplicate records.
3. Explore the dataset using descriptive statistics and visualizations.
4. Separate the input features from the target crop label.
5. Split the data into training and testing sets.
6. Train and evaluate multiple classification models.
7. Compare model performance using accuracy, macro precision, macro recall, macro F1-score, ROC-AUC (where calculated), and cross-validation.
8. Save the selected trained model and required metadata.
9. Load the saved files in Streamlit.
10. Collect user inputs, arrange them in the training feature order, apply scaling if needed, and predict a crop.

The notebook `main.ipynb` contains the analysis and model-training workflow. The Streamlit application `app.py` provides the user interface for prediction.

## 📊 Model Evaluation

The project notebook compares five models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

In the reported test-set comparison from this project, Random Forest achieved approximately **99.55% accuracy**. Treat this as a result from the current dataset split, not a guarantee of performance on new farms or real-world conditions. For a strong evaluation, avoid data leakage and validate on suitable independent data.

## 🌐 Deploying to Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. Open Streamlit Community Cloud and sign in.
3. Create a new app and select your repository, branch, and `app.py` as the main file.
4. Deploy the app.
5. Confirm that the required model `.pkl` files are available to the deployed app.

Deployment can fail if required model files are absent from the repository or if the scikit-learn version differs from the version used to save the model. For reliable model loading, use a compatible scikit-learn version and record the versions used during training.

## ⚠️ Limitations and Responsible Use

- A model prediction is not a guarantee that a crop will grow successfully.
- Actual crop suitability also depends on region, season, soil characteristics, irrigation, pests, seed variety, and other local factors.
- Model scores are estimated class probabilities, not guaranteed probabilities of agricultural success.
- Consult local agricultural experts and soil-testing services before making important planting decisions.
- The model's performance depends on the quality, coverage, and representativeness of the training dataset.

## 🔮 Possible Future Improvements

- Add soil-test file upload and batch predictions.
- Add regional and seasonal context.
- Include a trusted weather-data integration.
- Provide multilingual interface options.
- Evaluate the model on independent, geographically diverse data.
- Add unit tests and input-range validation based on the training dataset.

## 👨‍💻 Author

**Your Name**

Add your GitHub profile, LinkedIn profile, and contact details here.

## 📄 License

Choose and add a license before allowing others to reuse or redistribute this project. If no license is included, users do not automatically receive permission to reuse the code.
