import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import shap
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="HVAC Predictive Maintenance Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model, Data, and Preprocessor ---
# This function caches the loaded objects to improve performance
@st.cache_resource
def load_resources():
    """Loads the ML model, preprocessor, SHAP explainer, and historical data."""
    try:
        model = joblib.load('hvac_xgboost_model.joblib')
        preprocessor = joblib.load('hvac_preprocessor.joblib')
        data = pd.read_csv('hvac_maintenance_data.csv')
        
        # Create a SHAP explainer object
        # This uses the preprocessed training data's structure to create explanations
        X = data.drop(columns=['criticality', 'alert_id'])
        X_processed = preprocessor.transform(X)
        explainer = shap.TreeExplainer(model, X_processed)
        
        return model, preprocessor, explainer, data
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}. Make sure the necessary files (.joblib, .csv) are in the same directory.")
        return None, None, None, None

model, preprocessor, explainer, df = load_resources()

if model is None:
    st.stop()

# --- Helper Function for Data Generation (from notebook) ---
# This function simulates new, incoming data for the real-time tab
def generate_live_data(asset_types, fault_codes):
    """Generates a single row of synthetic data based on notebook logic."""
    data_point = {
        'asset_type': np.random.choice(asset_types),
        'age_months': np.random.randint(1, 180),
        'last_service_days_ago': np.random.randint(1, 730),
        'fault_code': np.random.choice(fault_codes),
        'pressure_psi': np.random.normal(150, 20),
        'temperature_celsius': np.random.normal(22, 4)
    }
    # Ensure service days < age days
    while data_point['last_service_days_ago'] >= data_point['age_months'] * 30:
        data_point['last_service_days_ago'] = np.random.randint(1, 730)
    return data_point

# --- Dashboard UI ---
st.title("HVAC Predictive Maintenance Dashboard ❄️")

# Use tabs for different sections of the dashboard
tab1, tab2, tab3 = st.tabs(["📊 Historical Analysis", "🔮 Criticality Prediction Tool", "📡 Real-Time Simulation"])


# ==============================================================================
# TAB 1: HISTORICAL ANALYSIS
# ==============================================================================
with tab1:
    st.header("Historical Alert Analysis")

    # --- Sidebar Filters ---
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
        # --- KPIs / Metrics ---
        total_alerts = filtered_df.shape[0]
        high_criticality_alerts = filtered_df[filtered_df['criticality'] == 'High'].shape[0]
        avg_asset_age = int(filtered_df['age_months'].mean())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts", f"{total_alerts}")
        col2.metric("High Criticality Alerts", f"{high_criticality_alerts}")
        col3.metric("Avg. Asset Age (Months)", f"{avg_asset_age}")
        
        st.markdown("---")

        # --- Visualizations ---
        col1, col2 = st.columns((6, 4)) # Make first column wider

        with col1:
            st.subheader("Alerts by Asset Type and Criticality")
            fig_bar = px.bar(
                filtered_df,
                x='asset_type',
                color='criticality',
                barmode='group',
                title="Count of Alerts",
                labels={'asset_type': 'Asset Type', 'count': 'Number of Alerts'},
                color_discrete_map={'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#636EFA'}
            )
            fig_bar.update_layout(xaxis_title="Asset Type", yaxis_title="Number of Alerts")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Criticality Distribution")
            fig_pie = px.pie(
                filtered_df,
                names='criticality',
                title="Overall Criticality",
                hole=0.3,
                color_discrete_map={'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#636EFA'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Sensor Reading Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig_pressure = px.histogram(filtered_df, x='pressure_psi', title='Pressure (PSI) Distribution', nbins=30)
            st.plotly_chart(fig_pressure, use_container_width=True)
        with col2:
            fig_temp = px.histogram(filtered_df, x='temperature_celsius', title='Temperature (°C) Distribution', nbins=30)
            st.plotly_chart(fig_temp, use_container_width=True)
            
        st.subheader("Filtered Data View")
        st.dataframe(filtered_df)

# ==============================================================================
# TAB 2: CRITICALITY PREDICTION TOOL
# ==============================================================================
with tab2:
    st.header("Predict Alert Criticality")
    st.markdown("Use the form below to get a real-time prediction and explanation for a new alert.")
    
    # --- Input Form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            asset_type = st.selectbox("Asset Type", options=df['asset_type'].unique())
            age_months = st.slider("Age of Asset (months)", 1, 240, 60)
            last_service_days_ago = st.slider("Days Since Last Service", 1, 730, 90)
        with col2:
            fault_code = st.selectbox("Fault Code", options=df['fault_code'].unique())
            pressure_psi = st.number_input("Pressure (PSI)", 50.0, 250.0, 150.0, 0.1)
            temperature_celsius = st.number_input("Temperature (°C)", 0.0, 50.0, 22.0, 0.1)
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            'asset_type': asset_type,
            'age_months': age_months,
            'last_service_days_ago': last_service_days_ago,
            'fault_code': fault_code,
            'pressure_psi': pressure_psi,
            'temperature_celsius': temperature_celsius
        }])

        # --- Prediction and Explanation ---
        processed_input = preprocessor.transform(input_data)
        prediction_class = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)
        
        # Get class labels from the model
        class_labels = model.classes_
        predicted_criticality = class_labels[prediction_class[0]]
        
        st.subheader("Prediction Result")
        st.metric("Predicted Criticality", predicted_criticality)

        # Display probabilities in a more readable format
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame(prediction_proba, columns=class_labels)
        st.dataframe(prob_df.style.format("{:.2%}"))

        # --- SHAP Explanation Plot ---
        st.subheader("Prediction Explanation (SHAP Analysis)")
        shap_values = explainer.shap_values(processed_input)

        # Create the force plot using st.pydeck_chart as a placeholder for HTML rendering
        shap_html = f"<head>{shap.getjs()}</head><body>{shap.force_plot(explainer.expected_value[prediction_class[0]], shap_values[prediction_class[0]], input_data, show=False)}</body>"
        st.components.v1.html(shap_html, height=150)
        st.info("The plot above shows the features pushing the prediction towards (red) or away from (blue) the final outcome. The 'base value' is the average prediction over the training data.")

# ==============================================================================
# TAB 3: REAL-TIME SIMULATION
# ==============================================================================
with tab3:
    st.header("Live Alert Feed Simulation")
    
    if 'live_data' not in st.session_state:
        st.session_state.live_data = pd.DataFrame(columns=list(df.columns.drop(['criticality', 'alert_id'])))

    # --- Control Panel ---
    is_running = st.toggle("Start/Stop Simulation", value=False)
    
    # --- Placeholders for live updates ---
    placeholder = st.empty()
    
    if is_running:
        while True:
            # Generate new data point
            new_data_point = generate_live_data(df['asset_type'].unique(), df['fault_code'].unique())
            new_df_row = pd.DataFrame([new_data_point])

            # Preprocess and predict
            processed_new = preprocessor.transform(new_df_row)
            prediction = model.predict(processed_new)
            new_data_point['criticality'] = model.classes_[prediction[0]]
            
            # Add to session state
            st.session_state.live_data = pd.concat([pd.DataFrame([new_data_point]), st.session_state.live_data], ignore_index=True).head(50) # Keep last 50
            
            with placeholder.container():
                # --- Live Metrics ---
                kpi1, kpi2, kpi3 = st.columns(3)
                
                # Highlight if the latest alert is 'High'
                latest_criticality = st.session_state.live_data.iloc[0]['criticality']
                if latest_criticality == 'High':
                    kpi1.error(f"🔴 Latest Alert: High Criticality")
                else:
                    kpi1.success(f"🟢 Latest Alert: {latest_criticality}")
                
                high_count = (st.session_state.live_data['criticality'] == 'High').sum()
                kpi2.metric("High Criticality Count (last 50)", f"{high_count}")
                kpi3.metric("Total Alerts Simulated", f"{len(st.session_state.live_data)}")

                # --- Live Graph ---
                st.subheader("Live Criticality Counts")
                fig_live_bar = px.bar(
                    st.session_state.live_data['criticality'].value_counts().reset_index(),
                    x='criticality',
                    y='count',
                    title="Live Distribution of Alert Criticality",
                    color_discrete_map={'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#636EFA'},
                    color='criticality'
                )
                st.plotly_chart(fig_live_bar, use_container_width=True)

                # --- Live Data Table ---
                st.subheader("Latest Alerts Log")
                st.dataframe(st.session_state.live_data)

            time.sleep(2) # Delay to simulate real-time feed