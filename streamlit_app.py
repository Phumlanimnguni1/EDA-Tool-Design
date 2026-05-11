import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# CONFIGURATION & HELPER FUNCTIONS
st.set_page_config(page_title="DataQuest 2026: Credit Risk EDA", layout="wide")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def calculate_woe_iv(df, feature, target):
    """Calculates Weight of Evidence (WoE) and Information Value (IV)."""
    # Create a copy and handle numerical binning if necessary
    temp_df = df[[feature, target]].copy()
    
    if pd.api.types.is_numeric_dtype(temp_df[feature]):
        # Bin continuous variables into deciles for WoE calculation
        temp_df['binned'] = pd.qcut(temp_df[feature], q=10, duplicates='drop').astype(str)
        col_to_use = 'binned'
    else:
        col_to_use = feature

    # Calculate Good/Bad counts
    grouped = temp_df.groupby(col_to_use)[target].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'Total', 'sum': 'Bad'})
    grouped['Good'] = grouped['Total'] - grouped['Bad']

    # Handle zero counts by adding a small epsilon (0.5) to avoid divide-by-zero
    grouped['Good'] = np.where(grouped['Good'] == 0, 0.5, grouped['Good'])
    grouped['Bad'] = np.where(grouped['Bad'] == 0, 0.5, grouped['Bad'])

    # Calculate distributions
    total_good = grouped['Good'].sum()
    total_bad = grouped['Bad'].sum()
    grouped['Dist_Good'] = grouped['Good'] / total_good
    grouped['Dist_Bad'] = grouped['Bad'] / total_bad

    # Calculate WoE and IV
    grouped['WoE'] = np.log(grouped['Dist_Good'] / grouped['Dist_Bad'])
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']
    
    iv_total = grouped['IV'].sum()
    return grouped[['Total', 'Good', 'Bad', 'WoE', 'IV']], iv_total

# --- UI & LAYOUT ---
st.title("🏦 Credit Risk EDA Explorer")
st.markdown("Interactive tool for identifying risk patterns and evaluating feature predictive power.")

# Sidebar for Data Upload and Settings
with st.sidebar:
    st.header("1. Load Data")
    uploaded_file = st.file_uploader("Upload Loan Book CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        
        st.header("2. Define Target")
        target_col = st.selectbox("Select Target Variable (e.g., Default_Flag)", options=df.columns)
    else:
        st.warning("Please upload a dataset to begin.")

# Main Application Logic
if uploaded_file is not None:
    # Create tabs for the required sections
    tab1, tab2, tab3 = st.tabs(["Data Quality Report", "Univariate Explorer (WoE/IV)", "Bivariate Explorer"])

    # --- TAB 1: DATA QUALITY REPORT ---
    with tab1:
        st.header("Data Quality Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Feature', 'Missing Count']
            missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
            st.dataframe(missing_df[missing_df['Missing Count'] > 0].style.format({'Missing %': '{:.2f}%'}))
            
        with col2:
            st.subheader("Target Distribution")
            fig = px.pie(df, names=target_col, title="Target Class Balance")
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

    # --- TAB 2: UNIVARIATE EXPLORER ---
    with tab2:
        st.header("Univariate Analysis & Predictive Power")
        feature_col = st.selectbox("Select Feature to Analyze", options=[c for c in df.columns if c != target_col])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Distribution")
            # Plot distribution differing across target
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                fig = px.histogram(df, x=feature_col, color=target_col, barmode="overlay", 
                                   title=f"Distribution of {feature_col} by Target")
            else:
                fig = px.histogram(df, x=feature_col, color=target_col, barmode="group",
                                   title=f"Count of {feature_col} by Target")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Information Value (IV) & WoE")
            try:
                # Calculate and display WoE / IV
                woe_df, total_iv = calculate_woe_iv(df, feature_col, target_col)
                st.metric(label="Total Information Value (IV)", value=f"{total_iv:.4f}")
                
                # Simple IV rule of thumb guide
                if total_iv < 0.02: interpretation = "Useless for prediction"
                elif total_iv < 0.1: interpretation = "Weak predictive power"
                elif total_iv < 0.3: interpretation = "Medium predictive power"
                elif total_iv < 0.5: interpretation = "Strong predictive power"
                else: interpretation = "Suspiciously high (check for leakage)"
                
                st.caption(f"**Interpretation:** {interpretation}")
                st.dataframe(woe_df.style.background_gradient(subset=['WoE'], cmap='RdYlGn'))
            except Exception as e:
                st.error("Could not calculate WoE/IV. Ensure the target variable is binary (0/1).")

    # --- TAB 3: BIVARIATE EXPLORER ---
    with tab3:
        st.header("Bivariate Relationships & Interactions")
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Select X-axis Feature", options=df.columns, index=0)
        with col2:
            y_col = st.selectbox("Select Y-axis Feature", options=df.columns, index=1)
            
        st.subheader("Scatterplot")
        fig_scatter = px.scatter(df, x=x_col, y=y_col, color=target_col, 
                                 opacity=0.6, title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Numerical Correlation Heatmap")
        # Only plot numeric columns for the heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto", 
                             color_continuous_scale='RdBu_r', title="Correlation Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)