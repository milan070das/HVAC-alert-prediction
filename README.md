# HVAC Predictive Maintenance❄️

This project is an end-to-end machine learning application designed to predict the criticality of maintenance alerts for HVAC (Heating, Ventilation, and Air Conditioning) systems. It features a trained XGBoost model and an interactive web dashboard built with Streamlit for data analysis, real-time predictions, and model explainability.

## 🚀 Features

The application is organized into three main tabs, providing a comprehensive toolkit for maintenance analysis:

*   **📊 Historical Analysis**: An interactive dashboard for exploring the historical maintenance alert data.
    *   **KPIs**: Key metrics such as Total Alerts, High-Criticality Alerts, and Average Asset Age.
    *   **Interactive Visualizations**: Bar charts, pie charts, and histograms created with Plotly to analyze alert distributions by asset type, criticality, and sensor readings.
    *   **Dynamic Filtering**: Users can filter the historical data by asset type and criticality level to drill down into specific areas of interest.

*   **🔮 Criticality Prediction Tool**: A user-friendly interface to get on-demand predictions for new maintenance alerts.
    *   **Input Form**: A simple form to input asset details, sensor readings, and fault codes.
    *   **Real-Time Prediction**: Utilizes a pre-trained XGBoost model to instantly predict the criticality level (High, Medium, or Low) and display the associated probabilities.
    *   **AI Explainability (XAI)**: Integrates SHAP (SHapley Additive exPlanations) to provide a force plot that explains which features contributed most to the prediction, making the model's decision process transparent.

*   **📡 Real-Time Simulation**: A live feed that simulates incoming HVAC alerts to monitor system health in real-time.
    *   **Live Data Generation**: Continuously generates new, synthetic alert data points.
    *   **Live Predictions & Metrics**: Predicts the criticality for each new alert and updates KPIs, including the latest alert status and a count of recent high-criticality events.
    *   **Live Dashboard**: Displays a constantly updating data log and a bar chart of live alert distributions.

## 🛠️ Technical Stack

*   **Machine Learning**: Scikit-learn, XGBoost, SHAP
*   **Data Handling**: Pandas, NumPy
*   **Web Framework**: Streamlit
*   **Data Visualization**: Plotly, Matplotlib
*   **Development Environment**: Jupyter Notebook, Python 3


## ⚙️ Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    The project uses a `requirements.txt` file to manage dependencies[3].
    ```
    pip install -r requirements.txt
    ```

## 📈 Usage

The project requires a two-step process to get started:

1.  **Train the Model**:
    First, you need to generate the dataset and train the model. Open and run all cells in the `model_train.ipynb` Jupyter Notebook. This script will:
    *   Generate the `hvac_maintenance_data.csv` file.
    *   Train the XGBoost model and save it as `hvac_xgboost_model.joblib`.
    *   Create and save the data preprocessor as `hvac_preprocessor.joblib`.

2.  **Run the Streamlit Application**:
    Once the necessary files (`.csv`, `.joblib`) are generated, launch the web application:
    ```
    streamlit run app.py
    ```
    The application will open in your default web browser.

## 🔬 How It Works

### 1. Data Generation
A high-fidelity synthetic dataset is generated using the `model_train.ipynb` notebook. This process involves creating realistic base data and then applying a set of rules and noise layers to produce features and a target variable (`criticality`) that mimic real-world scenarios.

### 2. Model Training
*   **Preprocessing**: Categorical features (`asset_type`, `fault_code`) are one-hot encoded, and numerical features are passed through using a `ColumnTransformer`.
*   **Classifier**: An **XGBoost Classifier** is trained on the preprocessed data to predict one of three criticality levels: High, Medium, or Low. The model achieves an accuracy of approximately 95% on the test set.

### 3. Model Explainability
To ensure the model is not a "black box," **SHAP** is used for explainability:
*   **Local Explanations**: For individual predictions in the "Prediction Tool," a SHAP force plot visualizes the features that push the model's output towards or away from the final prediction.
*   **Global Explanations**: The training notebook also generates global feature importance plots, providing an overview of which features are most influential across all predictions.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
