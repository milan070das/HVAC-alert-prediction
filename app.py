import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import shap
import time
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="HVAC Predictive Maintenance Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)
plt.style.use('default')

# --- Load All Necessary Assets ---
@st.cache_resource
def load_resources():
    """
    Loads the ML model, preprocessor, label encoder, feature names, 
    and the SHAP explainer from disk.
    """
    try:
        model = joblib.load('hvac_xgboost_model.joblib')
        preprocessor = joblib.load('hvac_preprocessor.joblib')
        label_encoder = joblib.load('hvac_label_encoder.joblib')
        feature_names = joblib.load('hvac_feature_names.joblib')
        explainer = joblib.load('hvac_shap_explainer.joblib')
        data = pd.read_csv('hvac_maintenance_data.csv')
        return model, preprocessor, label_encoder, feature_names, explainer, data
    except FileNotFoundError as e:
        st.error(f"Error loading required file: {e}. Please ensure all .joblib and .csv files from the training notebook are in the same directory.")
        return (None,) * 6

model, preprocessor, label_encoder, feature_names, explainer, df = load_resources()

if model is None:
    st.stop()

# --- Helper Functions ---
def create_shap_waterfall_plot(shap_explanation_slice):
    """
    Generates a SHAP waterfall plot as a base64 encoded image
    to be displayed in Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation_slice, max_display=15, show=False)
    fig.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # Close the plot to free up memory
    
    # Encode the image to base64
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

def generate_live_data(asset_types, fault_codes):
    """Generates a single row of synthetic data."""
    data_point = {
        'asset_type': np.random.choice(asset_types),
        'age_months': np.random.randint(1, 180),
        'last_service_days_ago': np.random.randint(1, 730),
        'fault_code': np.random.choice(fault_codes),
        'pressure_psi': np.random.normal(150, 20),
        'temperature_celsius': np.random.normal(22, 4)
    }
    # Ensure service days < age days
    if data_point['last_service_days_ago'] >= data_point['age_months'] * 30:
        data_point['last_service_days_ago'] = np.random.randint(1, data_point['age_months'] * 30)
    return data_point

# --- Dashboard UI ---
st.title("HVAC Predictive Maintenance Dashboard ❄️")
tab1, tab2, tab3 = st.tabs(["📊 Historical Analysis", "🔮 Criticality Prediction Tool", "📡 Real-Time Simulation"])

# ==============================================================================
# TAB 1: HISTORICAL ANALYSIS
# ==============================================================================
with tab1:
    st.header("Historical Alert Analysis")
    st.sidebar.header("Filter Historical Data")
    
    asset_type_filter = st.sidebar.multiselect(
        'Filter by Asset Type:',
        options=df['asset_type'].unique(),
        default=df['asset_type'].unique()
    )
    criticality_filter = st.sidebar.multiselect(
        'Filter by Criticality:',
        options=df['criticality'].unique(),
        default=df['criticality'].unique()
    )
    filtered_df = df[df['asset_type'].isin(asset_type_filter) & df['criticality'].isin(criticality_filter)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        # KPIs
        total_alerts, high_alerts, avg_age = st.columns(3)
        total_alerts.metric("Total Alerts", f"{filtered_df.shape[0]}")
        high_alerts.metric("High Criticality Alerts", f"{filtered_df[filtered_df['criticality'] == 'High'].shape[0]}")
        avg_age.metric("Avg. Asset Age (Months)", f"{int(filtered_df['age_months'].mean())}")
        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Alerts by Asset Type and Criticality")
            fig = px.bar(filtered_df.groupby(['asset_type', 'criticality']).size().reset_index(name='count'),
                         x='asset_type', y='count', color='criticality', barmode='group',
                         color_discrete_map={'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#636EFA'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Criticality Distribution")
            fig = px.pie(filtered_df, names='criticality', hole=0.4,
                         color_discrete_map={'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#636EFA'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Filtered Data View")
        st.dataframe(filtered_df)

# ==============================================================================
# TAB 2: CRITICALITY PREDICTION TOOL
# ==============================================================================
with tab2:
    st.header("Predict Alert Criticality")
    st.markdown("Use the form to get a real-time prediction and explanation for a new alert.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            asset_type = st.selectbox("Asset Type", options=df['asset_type'].unique())
            age_months = st.slider("Age of Asset (months)", 1, 240, 60)
            last_service_days_ago = st.slider("Days Since Last Service", 1, 1000, 90)
        with col2:
            fault_code = st.selectbox("Fault Code", options=df['fault_code'].unique())
            pressure_psi = st.number_input("Pressure (PSI)", 50.0, 250.0, 150.0, 0.1)
            temperature_celsius = st.number_input("Temperature (°C)", 0.0, 50.0, 22.0, 0.1)
        
        submitted = st.form_submit_button("Predict Criticality")

    if submitted:
        input_data = pd.DataFrame([{
            'asset_type': asset_type, 'age_months': age_months,
            'last_service_days_ago': last_service_days_ago, 'fault_code': fault_code,
            'pressure_psi': pressure_psi, 'temperature_celsius': temperature_celsius
        }])

        # Process input and predict
        processed_input = preprocessor.transform(input_data)
        prediction_index = model.predict(processed_input)[0]
        prediction_name = label_encoder.classes_[prediction_index]
        prediction_proba = model.predict_proba(processed_input)[0]

        st.subheader("Prediction Result")
        color = "green"
        if prediction_name == 'High': color = "red"
        elif prediction_name == 'Medium': color = "orange"
        st.markdown(f"Predicted Criticality: <strong style='color:{color}; font-size:1.2em;'>{prediction_name}</strong>", unsafe_allow_html=True)
        

        # SHAP Explanation Plot
        st.subheader("Prediction Explanation (SHAP Waterfall Plot)")
        try:
            # Create a SHAP Explanation object for the input
            shap_explanation = explainer(processed_input)
            
            # Slice the explanation for the specific predicted class
            instance_class_explanation = shap_explanation[0, :, prediction_index]
            
            # Generate and display the plot
            waterfall_html = create_shap_waterfall_plot(instance_class_explanation)
            st.markdown(waterfall_html, unsafe_allow_html=True)

            st.info("""
            **How to read this plot:**
            *   **E[f(x)]** is the baseline prediction score (the average for this class).
            *   Features in **red** push the score **higher** (increasing the likelihood of this prediction).
            *   Features in **blue** push the score **lower**.
            *   **f(x)** is the final output score for this specific prediction.
            """)
        except Exception as e:
            st.error(f"An error occurred while generating the SHAP explanation: {e}")

# ==============================================================================
# TAB 3: REAL-TIME SIMULATION
# ==============================================================================
with tab3:
    st.header("Live Alert Feed Simulation")
    if 'live_data' not in st.session_state:
        st.session_state.live_data = pd.DataFrame()

    is_running = st.toggle("Start/Stop Simulation", value=False)
    placeholder = st.empty()

    if is_running:
        new_data_point = generate_live_data(df['asset_type'].unique(), df['fault_code'].unique())
        new_df_row = pd.DataFrame([new_data_point])
        
        # Predict criticality
        processed_new = preprocessor.transform(new_df_row)
        prediction_index = model.predict(processed_new)[0]
        new_data_point['predicted_criticality'] = label_encoder.classes_[prediction_index]
        
        # Prepend to session state dataframe
        new_row_df = pd.DataFrame([new_data_point])
        st.session_state.live_data = pd.concat([new_row_df, st.session_state.live_data], ignore_index=True).head(20)

        with placeholder.container():
            kpi1, kpi2, kpi3 = st.columns(3)
            latest_criticality = st.session_state.live_data.iloc[0]['predicted_criticality']
            color = {'High': 'error', 'Medium': 'warning', 'Low': 'success'}.get(latest_criticality, 'info')
            getattr(kpi1, color)(f"🔴 Latest Alert: {latest_criticality}")

            high_count = (st.session_state.live_data['predicted_criticality'] == 'High').sum()
            kpi2.metric("High Criticality Alerts (last 20)", f"{high_count}")
            kpi3.metric("Total Alerts Simulated", len(st.session_state.live_data))

            st.subheader("Live Data Log")
            st.dataframe(st.session_state.live_data)
        
        time.sleep(2) # Wait 2 seconds before rerunning
        st.rerun()
