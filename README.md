# Predictive Maintenance for HVAC Alert Prioritization

This project demonstrates an end-to-end machine learning workflow for a predictive maintenance task. The goal is to build a model that automatically prioritizes maintenance alerts from HVAC (Heating, Ventilation, and Air Conditioning) systems by classifying their criticality. This helps maintenance teams manage "alert fatigue" and focus their resources on the most urgent issues, preventing costly equipment failures.

The entire process, from data generation to model interpretation, is contained within the `model_train.ipynb` notebook.

## Key Features

*   **High-Fidelity Synthetic Data Generation**: Instead of relying on pre-existing data, the project begins by generating a realistic, noise-injected synthetic dataset. This process programmatically defines a logical "ground truth" for alert criticality based on equipment age, service history, and fault type.
*   **High-Performance Predictive Model**: Utilizes an XGBoost classifier, a powerful gradient-boosting algorithm, to achieve **~95% accuracy** in predicting alert criticality (High, Medium, or Low).
*   **Explainable AI (XAI)**: Implements SHAP (SHapley Additive exPlanations) to make the model's decisions transparent. This provides both global feature importance summaries and local, instance-level explanations, building trust and making the model's output actionable for technicians.

---

## Workflow

The project follows these key steps, each documented in the Jupyter Notebook:

1.  **Synthetic Data Generation**: A Python function (`generate_ideal_hvac_data`) creates a dataset of 20,000 realistic HVAC alerts. It enforces logical consistency (e.g., service date cannot precede installation date) and introduces both measurement noise (for sensor readings) and outcome noise (for labeling) to better simulate real-world conditions. The final dataset is saved as `hvac_maintenance_data.csv`.

2.  **Data Preprocessing**: The generated data is loaded and prepared for modeling. This includes:
    *   Splitting the data into training (80%) and testing (20%) sets while stratifying by the target label to maintain class distribution.
    *   Applying one-hot encoding to categorical features (`asset_type`, `fault_code`).
    *   Using `LabelEncoder` to convert the text-based target variable (`criticality`) into numerical format.

3.  **Model Training**: An `XGBClassifier` is trained on the preprocessed training data.

4.  **Model Evaluation**: The model's performance is assessed on the unseen test data. It achieves an overall accuracy of **94.69%**, with particularly high precision and recall for 'High' and 'Medium' priority alerts.

5.  **Model Interpretation**: SHAP is used to explain the model's predictions. The analysis reveals:
    *   **Global Importance**: The most influential features driving predictions are `last_service_days_ago` and `age_months`.
    *   **Local Explanations**: Waterfall plots are generated to break down individual predictions, showing exactly how each feature contributed to the final criticality score.

---

## Technologies Used

*   Python
*   Jupyter Notebook
*   Pandas
*   NumPy
*   Scikit-learn
*   XGBoost
*   SHAP
*   Matplotlib

---

## How to Run

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    The notebook installs packages directly using `%pip`. You can run the first two cells of the notebook, or create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.

    **requirements.txt:**
    ```
    numpy
    pandas
    scikit-learn
    xgboost
    shap
    matplotlib
    ```

4.  **Run the Jupyter Notebook:**
    Launch Jupyter and open the `model_train.ipynb` file.
    ```
    jupyter notebook model_train.ipynb
    ```

5.  **Execute the cells:**
    Run the cells in the notebook sequentially. The first run will generate the `hvac_maintenance_data.csv` file, which is then used in subsequent cells for training and evaluation.

---
