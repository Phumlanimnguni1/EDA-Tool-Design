import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn
try:
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="DataQuest 2026: Credit Risk EDA", layout="wide", initial_sidebar_state="expanded")

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file):
    """Load and optimize data types for memory efficiency."""
    df = pd.read_csv(file)
    
    # Memory optimization - but keep float64 for compatibility
    for col in df.columns:
        if df[col].dtype == 'int64':
            # Check if we can downcast integers safely
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                    df[col] = df[col].astype('int32')
        # Keep float64 to avoid JSON serialization issues with plotly
            
    return df

def clean_categorical(series, min_frequency=0.01):
    """Group rare categories into 'Other' to prevent sparse WoE bins."""
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_categorical_dtype(series):
        return series
    
    value_counts = series.value_counts(normalize=True)
    rare_categories = value_counts[value_counts < min_frequency].index
    
    cleaned = series.copy()
    cleaned = cleaned.replace(rare_categories, 'Other_Rare')
    return cleaned

def calculate_woe_iv(df, feature, target, bins=10):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV).
    
    WoE = ln(% of Goods / % of Bads)
    IV = Σ (% of Goods - % of Bads) × WoE
    """
    temp_df = df[[feature, target]].copy().dropna()
    
    if len(temp_df) == 0:
        return pd.DataFrame(), 0.0
    
    # Handle numeric vs categorical
    if pd.api.types.is_numeric_dtype(temp_df[feature]):
        try:
            temp_df['binned'] = pd.qcut(temp_df[feature], q=bins, duplicates='drop')
            col_to_use = 'binned'
        except:
            # If qcut fails, use cut instead
            temp_df['binned'] = pd.cut(temp_df[feature], bins=bins, duplicates='drop')
            col_to_use = 'binned'
    else:
        # Clean categorical before grouping
        temp_df[feature] = clean_categorical(temp_df[feature])
        col_to_use = feature

    # Group and calculate
    grouped = temp_df.groupby(col_to_use, observed=True)[target].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'Total', 'sum': 'Bad'})
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    
    # Avoid division by zero
    grouped['Good'] = np.where(grouped['Good'] == 0, 0.5, grouped['Good'])
    grouped['Bad'] = np.where(grouped['Bad'] == 0, 0.5, grouped['Bad'])
    
    total_good = grouped['Good'].sum()
    total_bad = grouped['Bad'].sum()
    
    grouped['Dist_Good'] = grouped['Good'] / total_good
    grouped['Dist_Bad'] = grouped['Bad'] / total_bad
    grouped['Default_Rate'] = (grouped['Bad'] / grouped['Total']) * 100
    
    # Calculate WoE and IV
    grouped['WoE'] = np.log(grouped['Dist_Good'] / grouped['Dist_Bad'])
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']
    
    iv_total = grouped['IV'].sum()
    
    # Convert to standard Python types for JSON serialization
    result_df = grouped.reset_index()
    for col in result_df.columns:
        if pd.api.types.is_numeric_dtype(result_df[col]):
            result_df[col] = result_df[col].astype(float)
    
    return result_df, float(iv_total)

def interpret_iv(iv):
    """Provide interpretation and color coding for IV values."""
    if iv < 0.02:
        return "❌ Useless for prediction", "red", 0
    elif iv < 0.1:
        return "⚠️ Weak predictive power", "orange", 1
    elif iv < 0.3:
        return "✅ Medium predictive power", "blue", 2
    elif iv < 0.5:
        return "🔥 Strong predictive power", "green", 3
    else:
        return "⚠️ Suspiciously high (check for leakage)", "red", 4

def calculate_gini(y_true, y_pred):
    """Calculate Gini coefficient from predictions."""
    if SKLEARN_AVAILABLE:
        try:
            auc_score = roc_auc_score(y_true, y_pred)
            gini = 2 * auc_score - 1
            return gini
        except:
            return None
    else:
        # Manual calculation
        return calculate_gini_manual(y_true, y_pred)

def calculate_gini_manual(y_true, y_pred):
    """Manual Gini calculation without sklearn."""
    # Sort by prediction score
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # Calculate cumulative sums
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Count concordant pairs
    cum_pos = 0
    auc_sum = 0
    
    for yt in y_true_sorted:
        if yt == 1:
            cum_pos += 1
        else:
            auc_sum += cum_pos
    
    auc_score = auc_sum / (n_pos * n_neg)
    gini = 2 * auc_score - 1
    return gini

def safe_plotly_data(df):
    """Convert dataframe to safe types for Plotly JSON serialization."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(float)
    return df_copy

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- MAIN UI ---
st.markdown('<p class="main-header">🏦 DataQuest 2026: Credit Risk EDA Explorer</p>', unsafe_allow_html=True)
st.markdown("**Interactive tool for identifying risk patterns and evaluating feature predictive power in credit modeling.**")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bank-building.png", width=80)
    st.header("⚙️ Configuration")
    
    st.subheader("1️⃣ Load Data")
    uploaded_file = st.file_uploader("Upload Loan Book CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
        st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.info(f"💾 Memory: {mem_usage:.2f} MB")
        
        st.subheader("2️⃣ Define Target")
        binary_columns = [col for col in df.columns if df[col].nunique() == 2]
        
        if len(binary_columns) > 0:
            target_col = st.selectbox("Select Target Variable (0/1)", options=binary_columns, index=0)
            
            # Display key metrics
            default_rate = float(df[target_col].mean() * 100)
            st.metric("📊 Default Rate", f"{default_rate:.2f}%")
            st.metric("📈 Total Loans", f"{len(df):,}")
            st.metric("⚠️ Defaults", f"{int(df[target_col].sum()):,}")
            
        else:
            st.error("❌ No binary target column found.")
            st.stop()
            
        st.subheader("3️⃣ Settings")
        woe_bins = st.slider("WoE Binning (for numeric features)", 5, 20, 10)
        
    else:
        st.warning("⬆️ Please upload a dataset to begin.")
        st.stop()

# --- MAIN APPLICATION ---
if uploaded_file is not None:
    
    # Filter valid features (exclude target and high-cardinality categoricals)
    valid_features = []
    for col in df.columns:
        if col != target_col:
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_features.append(col)
            elif df[col].nunique() < 100:  # Reasonable cardinality for categoricals
                valid_features.append(col)
    
    # --- TABS ---
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Research & Concepts",
        "🔍 Data Quality Report", 
        "📊 Univariate Explorer (WoE/IV)",
        "🔗 Bivariate Explorer",
        "🏆 Feature Ranking"
    ])
    
    # ========================================
    # TAB 0: RESEARCH & CONCEPTS
    # ========================================
    with tab0:
        st.header("📚 Research: Credit Modeling Fundamentals")
        
        st.markdown("---")
        
        # Section 1: GLMs vs Non-Linear Models
        st.subheader("1. Generalized Linear Models vs Non-Linear Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### **Logistic Regression (GLM)**
            
            Logistic regression models the probability of default using a linear combination of features:
            
            **Mathematical Form:**
            """)
            st.latex(r"\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n")
            
            st.markdown("""
            Where:
            - $p$ = probability of default
            - $\\beta_i$ = coefficients (interpretable weights)
            - $x_i$ = features
            
            **Advantages:**
            - ✅ **Interpretable**: Each coefficient shows feature impact
            - ✅ **Regulatory approved**: Explainable to auditors
            - ✅ **Stable**: Less prone to overfitting
            - ✅ **Fast**: Quick to train and deploy
            
            **Disadvantages:**
            - ❌ Assumes linear relationship (in log-odds space)
            - ❌ Cannot capture complex interactions automatically
            - ❌ Lower predictive power than ensemble methods
            """)
        
        with col2:
            st.markdown("""
            #### **Non-Linear Models (Random Forest, XGBoost, Neural Networks)**
            
            These models can capture complex, non-linear patterns through:
            - Decision trees (Random Forest, XGBoost)
            - Neural network layers (Deep Learning)
            
            **Advantages:**
            - ✅ **Higher accuracy**: Can model complex patterns
            - ✅ **Automatic interactions**: Finds feature combinations
            - ✅ **Flexible**: Handles non-linear relationships
            
            **Disadvantages:**
            - ❌ **Black box**: Difficult to explain predictions
            - ❌ **Regulatory concerns**: Hard to justify to auditors
            - ❌ **Overfitting risk**: Can memorize training data
            - ❌ **Computationally expensive**: Slower to train
            
            **Example Decision Boundary:**
            """)
            
            # Simple visualization
            fig = go.Figure()
            x = np.linspace(-3, 3, 100)
            y_linear = 1 / (1 + np.exp(-x))
            y_nonlinear = 1 / (1 + np.exp(-x**2))
            
            fig.add_trace(go.Scatter(x=x, y=y_linear, name="Logistic (Linear)", line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=x, y=y_nonlinear, name="Non-Linear Model", line=dict(color='red', width=3, dash='dash')))
            fig.update_layout(title="Decision Boundary Comparison", xaxis_title="Feature Value", yaxis_title="P(Default)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Section 2: Interpretability vs Complexity
        st.subheader("2. The Interpretability-Complexity Trade-off")
        
        st.markdown("""
        In credit modeling, **interpretability is often more valuable than marginal accuracy gains**.
        
        **Why Interpretability Matters in Credit:**
        - 🏛️ **Regulatory Compliance**: Regulators require explainable models (e.g., GDPR "right to explanation")
        - ⚖️ **Fair Lending Laws**: Must prove models don't discriminate
        - 🤝 **Customer Trust**: Applicants deserve to know why they were rejected
        - 🔍 **Model Validation**: Risk teams need to audit and validate logic
        - 🛡️ **Risk Management**: Understand what drives defaults
        """)
        
        # Trade-off visualization
        models = ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
        interpretability = [95, 90, 70, 40, 30, 20]
        complexity = [20, 25, 50, 75, 85, 95]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=complexity, y=interpretability, mode='markers+text', 
                                 marker=dict(size=15, color=interpretability, colorscale='RdYlGn', showscale=True),
                                 text=models, textposition="top center", textfont=dict(size=10)))
        fig.update_layout(title="Model Interpretability vs Complexity Trade-off",
                         xaxis_title="Model Complexity", yaxis_title="Interpretability Score",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Section 3: Credit Modeling Concepts
        st.subheader("3. Credit Modeling Concepts: WoE and IV")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### **Weight of Evidence (WoE)**
            
            WoE measures the strength of a feature in separating good vs bad customers.
            """)
            st.latex(r"WoE = \ln\left(\frac{\% \text{ of Goods}}{\% \text{ of Bads}}\right)")
            
            st.markdown("""
            **Interpretation:**
            - **Positive WoE**: More good customers than bad (lower risk)
            - **Negative WoE**: More bad customers than good (higher risk)
            - **WoE ≈ 0**: Neutral (no predictive power)
            
            **Why WoE is Useful:**
            - Transforms categorical variables into continuous scores
            - Handles missing values naturally
            - Creates monotonic relationships with target
            - Reduces impact of outliers
            """)
        
        with col2:
            st.markdown("""
            #### **Information Value (IV)**
            
            IV quantifies the overall predictive power of a feature.
            """)
            st.latex(r"IV = \sum (\% \text{ Goods} - \% \text{ Bads}) \times WoE")
            
            st.markdown("""
            **IV Interpretation Guidelines:**
            
            | IV Range | Predictive Power | Action |
            |----------|------------------|--------|
            | < 0.02 | Useless | ❌ Exclude |
            | 0.02 - 0.10 | Weak | ⚠️ Consider excluding |
            | 0.10 - 0.30 | Medium | ✅ Include |
            | 0.30 - 0.50 | Strong | ✅ Definitely include |
            | > 0.50 | Suspicious | ⚠️ Check for data leakage |
            
            **Why IV is Useful:**
            - Quick feature selection tool
            - Language-agnostic (works across all data types)
            - Industry standard in credit scoring
            """)
        
        st.markdown("---")
        
        # Section 4: Model Evaluation Metrics
        st.subheader("4. Key Evaluation Metrics for Credit Models")
        
        st.markdown("""
        Credit models require specialized metrics beyond simple accuracy because:
        - Classes are imbalanced (defaults are rare)
        - False positives and false negatives have different costs
        - Ranking quality matters (who gets approved first)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### **Confusion Matrix Metrics**
            
            |  | Predicted: No Default | Predicted: Default |
            |---|---|---|
            | **Actual: No Default** | True Negative (TN) | False Positive (FP) |
            | **Actual: Default** | False Negative (FN) | True Positive (TP) |
            
            **Accuracy:**
            """)
            st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
            st.markdown("⚠️ **Problem**: Misleading with imbalanced data (e.g., 95% accuracy by predicting 'no default' for everyone)")
            
            st.markdown("""
            **Precision (Positive Predictive Value):**
            """)
            st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
            st.markdown("📊 **Credit Context**: Of all loans we predicted would default, what % actually defaulted?")
            st.markdown("💡 **Business Impact**: High precision = fewer false alarms, less wasted investigation effort")
            
            st.markdown("""
            **Recall (Sensitivity, True Positive Rate):**
            """)
            st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
            st.markdown("📊 **Credit Context**: Of all loans that actually defaulted, what % did we catch?")
            st.markdown("💡 **Business Impact**: High recall = fewer bad loans slip through")
            
            st.markdown("""
            **F1 Score (Harmonic Mean):**
            """)
            st.latex(r"F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
            st.markdown("⚖️ Balances precision and recall when both matter equally")
        
        with col2:
            st.markdown("""
            #### **Ranking Metrics (Most Important for Credit)**
            
            **AUC (Area Under ROC Curve):**
            - Measures model's ability to rank risky customers higher than safe ones
            - Range: 0.5 (random) to 1.0 (perfect)
            - **Credit Benchmark**: 
              - AUC < 0.6: Poor
              - AUC 0.6-0.7: Fair
              - AUC 0.7-0.8: Good
              - AUC 0.8-0.9: Excellent
              - AUC > 0.9: Suspicious (possible overfitting)
            
            **Gini Coefficient:**
            """)
            st.latex(r"\text{Gini} = 2 \times AUC - 1")
            st.markdown("""
            - Range: 0 (random) to 1 (perfect)
            - Preferred in European credit markets
            - **Interpretation**: % improvement over random ranking
            
            **Why AUC/Gini Matter in Credit:**
            - ✅ Threshold-independent (works at any approval rate)
            - ✅ Robust to class imbalance
            - ✅ Measures ranking quality (who's riskiest?)
            - ✅ Industry standard for model comparison
            """)
            
            # ROC Curve example
            if SKLEARN_AVAILABLE:
                # Simulate predictions
                np.random.seed(42)
                y_true = np.concatenate([np.ones(200), np.zeros(800)])
                y_pred_good = np.concatenate([np.random.beta(6, 2, 200), np.random.beta(2, 6, 800)])
                y_pred_bad = np.concatenate([np.random.beta(3, 3, 200), np.random.beta(3, 3, 800)])
                
                fpr_good, tpr_good, _ = roc_curve(y_true, y_pred_good)
                fpr_bad, tpr_bad, _ = roc_curve(y_true, y_pred_bad)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr_good.tolist(), y=tpr_good.tolist(), 
                                         name=f'Good Model (AUC={auc(fpr_good, tpr_good):.3f})', 
                                         line=dict(color='green', width=3)))
                fig.add_trace(go.Scatter(x=fpr_bad.tolist(), y=tpr_bad.tolist(), 
                                         name=f'Poor Model (AUC={auc(fpr_bad, tpr_bad):.3f})', 
                                         line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random (AUC=0.5)', 
                                         line=dict(color='gray', width=2, dash='dash')))
                fig.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", 
                                 yaxis_title="True Positive Rate", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Install scikit-learn to see ROC curve visualization: `pip install scikit-learn`")
        
        st.markdown("---")
        
        # Section 5: Regulatory Concerns
        st.subheader("5. Regulatory Considerations & Prohibited Features")
        
        st.markdown("""
        Credit models must comply with fair lending laws and anti-discrimination regulations.
        
        #### **Protected Characteristics (Prohibited in Most Jurisdictions)**
        
        These features **cannot** be used directly in credit models:
        
        | Feature | Why Prohibited | Regulation |
        |---------|---------------|------------|
        | 🚫 **Race/Ethnicity** | Discrimination | Equal Credit Opportunity Act (ECOA), Fair Housing Act |
        | 🚫 **Gender/Sex** | Discrimination | ECOA, EU Gender Directive |
        | 🚫 **Age** (with exceptions) | Age discrimination | Age Discrimination Act (but can use for seniors 62+) |
        | 🚫 **Marital Status** | Discrimination | ECOA |
        | 🚫 **Religion** | Discrimination | Civil Rights Act |
        | 🚫 **National Origin** | Discrimination | ECOA |
        | 🚫 **Zip Code** (in some cases) | Proxy for race (redlining) | Fair Housing Act |
        | 🚫 **Neighborhood Demographics** | Proxy discrimination | Disparate Impact Doctrine |
        
        #### **Proxy Discrimination**
        
        Even if you don't use protected features directly, models can still discriminate through **proxy variables**:
        
        - **Example**: Using "email domain" might correlate with ethnicity if certain communities prefer specific providers
        - **Example**: "Years at current address" might disadvantage immigrants
        - **Example**: "Type of phone" (landline vs mobile) might correlate with age
        
        #### **Regulatory Requirements**
        
        1. **Model Documentation**: Must explain how model works
        2. **Adverse Action Notices**: Must tell rejected applicants why (top reasons)
        3. **Disparate Impact Testing**: Must prove model doesn't disproportionately reject protected groups
        4. **Model Validation**: Independent review required
        5. **Fair Lending Audits**: Regular testing for bias
        
        #### **In This Dataset**
        
        Potentially problematic features to watch:
        """)
        
        # Check for problematic features
        problematic = []
        feature_checks = {
            'age': '⚠️ Age - May require justification (but often allowed for credit risk)',
            'region': '⚠️ Region - Could be proxy for demographics (check for disparate impact)',
            'email_domain_type': '⚠️ Email Domain - Could correlate with protected classes',
            'home_ownership': '✅ Generally acceptable (economic factor, not protected class)',
            'employment_length': '✅ Generally acceptable (economic factor)'
        }
        
        for feature, note in feature_checks.items():
            if feature in df.columns:
                problematic.append(f"- **{feature}**: {note}")
        
        if problematic:
            st.markdown("\n".join(problematic))
        
        st.info("""
        💡 **Best Practice**: Always conduct disparate impact analysis and document business justification for every feature.
        Consult with legal/compliance teams before deploying credit models.
        """)
    
    # ========================================
    # TAB 1: DATA QUALITY REPORT
    # ========================================
    with tab1:
        st.header("🔍 Data Quality & Integrity Checks")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Values", f"{missing_pct:.2f}%")
        with col3:
            duplicate_count = df.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicate_count:,}")
        with col4:
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
            st.metric("Constant Features", len(constant_cols))
        
        st.markdown("---")
        
        # Row 1: Missing Values and Data Types
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Missing Values Analysis")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Feature', 'Missing Count']
            missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            if len(missing_df) > 0:
                # Convert to safe types
                missing_plot = safe_plotly_data(missing_df)
                fig = px.bar(missing_plot, x='Feature', y='Missing %', 
                            title="Missing Value Percentage by Feature",
                            color='Missing %', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(missing_df.style.format({'Missing %': '{:.2f}%'}), use_container_width=True)
            else:
                st.success("✅ No missing values found in dataset!")
        
        with col2:
            st.subheader("🔢 Data Types Distribution")
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)
            
            fig = px.pie(dtype_counts, values='Count', names='Data Type', 
                        title="Feature Data Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(dtype_counts, use_container_width=True)
        
        st.markdown("---")
        
        # Row 2: Categorical Inconsistencies
        st.subheader("📝 Text & Formatting Inconsistencies")
        
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        inconsistency_data = []
        
        for col in string_cols:
            non_null_series = df[col].dropna().astype(str)
            if non_null_series.empty:
                continue
            
            # Check for whitespace issues
            space_issues = non_null_series.str.contains(r'^\s+|\s+$', regex=True).sum()
            
            # Check for case inconsistencies
            original_unique = non_null_series.nunique()
            lower_unique = non_null_series.str.lower().nunique()
            case_issues = original_unique - lower_unique
            
            if space_issues > 0 or case_issues > 0:
                inconsistency_data.append({
                    "Feature": col,
                    "Unique Values": int(original_unique),
                    "Hidden Spaces": int(space_issues),
                    "Case Variations": int(case_issues),
                    "Recommendation": "Strip & standardize" if (space_issues > 0 and case_issues > 0) 
                                     else ("Strip spaces" if space_issues > 0 else "Standardize casing")
                })
        
        if inconsistency_data:
            st.warning("⚠️ Found categorical columns with formatting inconsistencies that may impact model performance.")
            inconsistency_df = pd.DataFrame(inconsistency_data)
            st.dataframe(inconsistency_df, use_container_width=True)
            
            # Show example for first problematic column
            if len(inconsistency_data) > 0:
                example_col = inconsistency_data[0]['Feature']
                st.info(f"**Example from '{example_col}'**: {df[example_col].value_counts().head(10).to_dict()}")
        else:
            st.success("✅ No formatting inconsistencies detected in categorical columns.")
        
        st.markdown("---")
        
        # Row 3: Outliers and Target Balance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Outlier Detection (IQR Method)")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_data = []
            
            for col in numeric_cols:
                Q1 = float(df[col].quantile(0.25))
                Q3 = float(df[col].quantile(0.75))
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    outlier_data.append({
                        "Feature": col,
                        "Outlier Count": len(outliers),
                        "Percentage": f"{(len(outliers)/len(df))*100:.2f}%",
                        "Lower Bound": f"{lower_bound:.2f}",
                        "Upper Bound": f"{upper_bound:.2f}"
                    })
            
            if outlier_data:
                outlier_df = pd.DataFrame(outlier_data).sort_values('Outlier Count', ascending=False)
                st.dataframe(outlier_df, use_container_width=True)
                st.caption("💡 Note: Outliers in credit data often represent legitimate extreme cases (high earners, large loans) rather than errors.")
            else:
                st.success("✅ No extreme outliers detected using IQR method.")
        
        with col2:
            st.subheader("🎯 Target Variable Distribution")
            
            target_counts = df[target_col].value_counts()
            target_pct = df[target_col].value_counts(normalize=True) * 100
            
            fig = go.Figure(data=[
                go.Bar(name='Count', x=[str(x) for x in target_counts.index], 
                      y=target_counts.values.tolist(), 
                      marker_color=['green', 'red'])
            ])
            fig.update_layout(title=f"Target Distribution: {target_col}", 
                            xaxis_title="Class", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Imbalance check
            imbalance_ratio = target_pct.max() / target_pct.min()
            if imbalance_ratio > 3:
                st.warning(f"⚠️ Class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
                st.info("💡 Consider using stratified sampling, SMOTE, or class weights in modeling.")
            else:
                st.success("✅ Classes are reasonably balanced.")
        
        st.markdown("---")
        
        # Row 4: Correlation Analysis
        st.subheader("🔗 Feature Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            # Find high correlations (excluding diagonal)
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = float(corr_matrix.iloc[i, j])
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Convert correlation matrix to float for plotly
                corr_matrix_float = corr_matrix.astype(float)
                fig = px.imshow(corr_matrix_float, 
                               text_auto='.2f',
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title="Feature Correlation Heatmap",
                               zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if high_corr:
                    st.warning(f"⚠️ Found {len(high_corr)} highly correlated feature pairs (|r| > 0.7)")
                    high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(high_corr_df.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
                    st.info("💡 Consider removing one feature from each pair to reduce multicollinearity.")
                else:
                    st.success("✅ No severe multicollinearity detected.")
        
        st.markdown("---")
        
                # Row 5: Summary Statistics
        st.subheader("📈 Summary Statistics")
        
        summary_stats = df.describe(include='all').T
        summary_stats['missing'] = df.isnull().sum()
        summary_stats['missing_pct'] = (df.isnull().sum() / len(df)) * 100
        
        st.dataframe(summary_stats.style.format({
            'missing_pct': '{:.2f}%',
            'mean': '{:.2f}',
            'std': '{:.2f}',
            'min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'max': '{:.2f}'
        }), use_container_width=True)
    
    # ========================================
    # TAB 2: UNIVARIATE EXPLORER
    # ========================================
    with tab2:
        st.header("📊 Univariate Analysis & Predictive Power")
        
        st.markdown("""
        Explore individual features to understand their distribution and relationship with the target variable.
        **Weight of Evidence (WoE)** and **Information Value (IV)** help identify which features are most predictive.
        """)
        
        # Feature selection
        feature_col = st.selectbox("🔍 Select Feature to Analyze", options=valid_features, key='univariate_feature')
        
        # Calculate WoE/IV
        with st.spinner("Calculating WoE and IV..."):
            woe_df, total_iv = calculate_woe_iv(df, feature_col, target_col, bins=woe_bins)
        
        # Display IV metric prominently
        interpretation, color, _ = interpret_iv(total_iv)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("📊 Information Value (IV)", f"{total_iv:.4f}")
        with col2:
            st.markdown(f"<div style='padding: 1rem; background-color: {color}20; border-left: 4px solid {color}; border-radius: 0.5rem; margin-top: 0.5rem;'><strong>Interpretation:</strong><br>{interpretation}</div>", unsafe_allow_html=True)
        with col3:
            st.info("""
            **IV Guidelines:**
            - < 0.02: Useless
            - 0.02-0.10: Weak
            - 0.10-0.30: Medium
            - 0.30-0.50: Strong
            - > 0.50: Suspicious (check for leakage)
            """)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Feature Distribution by Target")
            
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                # Histogram for numeric features
                fig = px.histogram(df, x=feature_col, color=target_col, 
                                  barmode="overlay", 
                                  title=f"Distribution of {feature_col}",
                                  opacity=0.7,
                                  color_discrete_map={0: 'green', 1: 'red'})
                fig.update_layout(legend_title_text='Default')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Bar chart for categorical features
                fig = px.histogram(df, x=feature_col, color=target_col,
                                  barmode="group",
                                  title=f"Count of {feature_col} by Target",
                                  color_discrete_map={0: 'green', 1: 'red'})
                fig.update_layout(legend_title_text='Default', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Default Rate by Bin")
            
            if len(woe_df) > 0:
                # Create bin labels
                if pd.api.types.is_numeric_dtype(df[feature_col]):
                    bin_labels = [str(x) for x in woe_df.iloc[:, 0]]
                else:
                    bin_labels = woe_df.iloc[:, 0].astype(str)
                
                # Convert to safe types
                woe_plot = woe_df.copy()
                woe_plot['bin_label'] = bin_labels
                woe_plot['Default_Rate'] = woe_plot['Default_Rate'].astype(float)
                
                fig = px.bar(woe_plot, x='bin_label', y='Default_Rate',
                            title=f"Default Rate by {feature_col} Bin",
                            labels={'bin_label': feature_col, 'Default_Rate': 'Default Rate (%)'},
                            color='Default_Rate',
                            color_continuous_scale='Reds')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to calculate default rates.")
        
        st.markdown("---")
        
        # WoE Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚖️ Weight of Evidence (WoE) Pattern")
            
            if len(woe_df) > 0:
                if pd.api.types.is_numeric_dtype(df[feature_col]):
                    bin_labels = [str(x) for x in woe_df.iloc[:, 0]]
                else:
                    bin_labels = woe_df.iloc[:, 0].astype(str)
                
                woe_values = woe_df['WoE'].astype(float).tolist()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bin_labels,
                    y=woe_values,
                    marker_color=['green' if x > 0 else 'red' for x in woe_values],
                    text=[f"{x:.3f}" for x in woe_values],
                    textposition='outside'
                ))
                fig.update_layout(
                    title=f"WoE Pattern for {feature_col}",
                    xaxis_title=feature_col,
                    yaxis_title="Weight of Evidence",
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **WoE Interpretation:**
                - 🟢 **Positive WoE**: Lower risk (more goods than bads)
                - 🔴 **Negative WoE**: Higher risk (more bads than goods)
                - **Monotonic trend**: Ideal for logistic regression
                """)
            else:
                st.warning("Unable to calculate WoE.")
        
        with col2:
            st.subheader("📊 Detailed WoE/IV Table")
            
            if len(woe_df) > 0:
                # Format the dataframe for display
                display_df = woe_df.copy()
                display_df = display_df.round({
                    'Total': 0,
                    'Good': 0,
                    'Bad': 0,
                    'Dist_Good': 4,
                    'Dist_Bad': 4,
                    'Default_Rate': 2,
                    'WoE': 4,
                    'IV': 4
                })
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download WoE Table",
                    data=csv,
                    file_name=f"woe_table_{feature_col}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No WoE data available.")
    
    # ========================================
    # TAB 3: BIVARIATE EXPLORER
    # ========================================
    with tab3:
        st.header("🔗 Bivariate Relationships & Interactions")
        
        st.markdown("""
        Explore relationships between two features and how they jointly relate to the target variable.
        This helps identify interaction effects and feature combinations.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Select X-axis Feature", options=valid_features, index=0, key='bivariate_x')
        with col2:
            y_col = st.selectbox("Select Y-axis Feature", options=valid_features, index=min(1, len(valid_features)-1), key='bivariate_y')
        
        st.markdown("---")
        
        # Check if both features are numeric
        x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
        
        if x_is_numeric and y_is_numeric:
            # Both numeric: scatter plot and density
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔵 Scatter Plot by Target")
                sample_df = df.sample(min(10000, len(df)))
                fig = px.scatter(
                    sample_df,
                    x=x_col, y=y_col, color=target_col,
                    opacity=0.6,
                    color_discrete_map={0: 'green', 1: 'red'},
                    title=f"{y_col} vs {x_col}",
                    render_mode='webgl'
                )
                fig.update_layout(legend_title_text='Default')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🌡️ Density Heatmap")
                fig = px.density_heatmap(
                    df, x=x_col, y=y_col,
                    facet_col=target_col,
                    title=f"Density: {y_col} vs {x_col}",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
            correlation = float(df[[x_col, y_col]].corr().iloc[0, 1])
            st.metric("📊 Correlation Coefficient", f"{correlation:.4f}")
            
            if abs(correlation) > 0.7:
                st.warning(f"⚠️ High correlation detected ({correlation:.3f}). Consider removing one feature to reduce multicollinearity.")
            
        elif not x_is_numeric and not y_is_numeric:
            # Both categorical: grouped bar chart
            st.subheader("📊 Default Rate by Feature Combination")
            
            # Calculate default rates for combinations
            grouped = df.groupby([x_col, y_col])[target_col].agg(['sum', 'count']).reset_index()
            grouped['default_rate'] = (grouped['sum'] / grouped['count']) * 100
            grouped = grouped[grouped['count'] >= 10]  # Filter small groups
            
            if len(grouped) > 0:
                # Convert to safe types
                grouped_plot = safe_plotly_data(grouped)
                fig = px.bar(grouped_plot, x=x_col, y='default_rate', color=y_col,
                            barmode='group',
                            title=f"Default Rate by {x_col} and {y_col}",
                            labels={'default_rate': 'Default Rate (%)'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                st.dataframe(grouped.style.format({'default_rate': '{:.2f}%'}), use_container_width=True)
            else:
                st.warning("Not enough data for meaningful analysis.")
        
        else:
            # Mixed: box plot or violin plot
            if x_is_numeric:
                numeric_col, cat_col = x_col, y_col
            else:
                numeric_col, cat_col = y_col, x_col
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📦 Box Plot by Category")
                fig = px.box(df, x=cat_col, y=numeric_col, color=target_col,
                            title=f"{numeric_col} Distribution by {cat_col}",
                            color_discrete_map={0: 'green', 1: 'red'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎻 Violin Plot by Category")
                fig = px.violin(df, x=cat_col, y=numeric_col, color=target_col,
                               box=True,
                               title=f"{numeric_col} Distribution by {cat_col}",
                               color_discrete_map={0: 'green', 1: 'red'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation heatmap for all numeric features
        st.subheader("🔥 Overall Correlation Heatmap")
        
        numeric_df = df[valid_features].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().astype(float)
            
            fig = px.imshow(corr_matrix,
                           text_auto='.2f',
                           aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title="Feature Correlation Matrix",
                           zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation analysis.")
    
    # ========================================
    # TAB 4: FEATURE RANKING
    # ========================================
    with tab4:
        st.header("🏆 Feature Ranking by Predictive Power")
        
        st.markdown("""
        This section ranks all features by their **Information Value (IV)** to help you identify
        the most predictive features for your credit model.
        """)
        
        # Calculate IV for all features
        with st.spinner("Calculating IV for all features... This may take a moment."):
            iv_summary = []
            
            progress_bar = st.progress(0)
            for idx, col in enumerate(valid_features):
                try:
                    _, iv = calculate_woe_iv(df, col, target_col, bins=woe_bins)
                    interpretation, color, rank = interpret_iv(iv)
                    iv_summary.append({
                        "Feature": col,
                        "IV": float(iv),
                        "Interpretation": interpretation,
                        "Rank": rank
                    })
                except Exception as e:
                    st.warning(f"Could not calculate IV for {col}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(valid_features))
            
            progress_bar.empty()
        
        if iv_summary:
            iv_df = pd.DataFrame(iv_summary).sort_values('IV', ascending=False)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                strong_features = len(iv_df[iv_df['IV'] >= 0.3])
                st.metric("🔥 Strong Features", strong_features, help="IV >= 0.3")
            with col2:
                medium_features = len(iv_df[(iv_df['IV'] >= 0.1) & (iv_df['IV'] < 0.3)])
                st.metric("✅ Medium Features", medium_features, help="0.1 <= IV < 0.3")
            with col3:
                weak_features = len(iv_df[(iv_df['IV'] >= 0.02) & (iv_df['IV'] < 0.1)])
                st.metric("⚠️ Weak Features", weak_features, help="0.02 <= IV < 0.1")
            with col4:
                useless_features = len(iv_df[iv_df['IV'] < 0.02])
                st.metric("❌ Useless Features", useless_features, help="IV < 0.02")
            
            st.markdown("---")
            
            # Visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Information Value by Feature")
                
                # Color code by interpretation
                colors = []
                for iv in iv_df['IV']:
                    if iv >= 0.5:
                        colors.append('red')
                    elif iv >= 0.3:
                        colors.append('green')
                    elif iv >= 0.1:
                        colors.append('blue')
                    elif iv >= 0.02:
                        colors.append('orange')
                    else:
                        colors.append('gray')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=iv_df['IV'].tolist(),
                    y=iv_df['Feature'].tolist(),
                    orientation='h',
                    marker_color=colors,
                    text=[f"{x:.4f}" for x in iv_df['IV']],
                    textposition='outside'
                ))
                
                # Add reference lines
                fig.add_vline(x=0.02, line_dash="dash", line_color="gray", annotation_text="Weak threshold")
                fig.add_vline(x=0.1, line_dash="dash", line_color="blue", annotation_text="Medium threshold")
                fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Strong threshold")
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Suspicious threshold")
                
                fig.update_layout(
                    title="Feature Ranking by Information Value",
                    xaxis_title="Information Value (IV)",
                    yaxis_title="Feature",
                    height=max(400, len(iv_df) * 25),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 IV Distribution")
                
                iv_categories = iv_df['Interpretation'].value_counts()
                fig = px.pie(values=iv_categories.values.tolist(), 
                            names=iv_categories.index.tolist(),
                            title="Features by Predictive Power")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed table
            st.subheader("📄 Detailed Feature Ranking Table")
            
            # Format the table
            display_df = iv_df.copy()
            display_df['IV'] = display_df['IV'].round(4)
            
                       # Detailed table
            st.subheader("📄 Detailed Feature Ranking Table")
            
            # Format the table
            display_df = iv_df.copy()
            display_df['IV'] = display_df['IV'].round(4)
            
            # Add color coding using apply instead of applymap
            def highlight_iv(row):
                iv_val = row['IV']
                if iv_val >= 0.5:
                    color = 'background-color: #ffcccc'
                elif iv_val >= 0.3:
                    color = 'background-color: #ccffcc'
                elif iv_val >= 0.1:
                    color = 'background-color: #cce5ff'
                elif iv_val >= 0.02:
                    color = 'background-color: #ffe5cc'
                else:
                    color = 'background-color: #e6e6e6'
                return ['' if col != 'IV' else color for col in row.index]
            
            styled_df = display_df.style.apply(highlight_iv, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feature Ranking",
                data=csv,
                file_name="feature_ranking_iv.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Recommendations
            st.subheader("💡 Feature Selection Recommendations")
            
            strong = iv_df[iv_df['IV'] >= 0.3]['Feature'].tolist()
            medium = iv_df[(iv_df['IV'] >= 0.1) & (iv_df['IV'] < 0.3)]['Feature'].tolist()
            weak = iv_df[(iv_df['IV'] >= 0.02) & (iv_df['IV'] < 0.1)]['Feature'].tolist()
            useless = iv_df[iv_df['IV'] < 0.02]['Feature'].tolist()
            suspicious = iv_df[iv_df['IV'] >= 0.5]['Feature'].tolist()
            
            if suspicious:
                st.error(f"⚠️ **Suspicious Features (IV > 0.5)**: {', '.join(suspicious)}")
                st.markdown("These features may indicate **data leakage**. Verify they don't contain information from the future or the target itself.")
            
            if strong:
                st.success(f"🔥 **Strong Features (0.3 ≤ IV < 0.5)**: {', '.join(strong)}")
                st.markdown("**Recommendation**: Definitely include these in your model. They have strong predictive power.")
            
            if medium:
                st.info(f"✅ **Medium Features (0.1 ≤ IV < 0.3)**: {', '.join(medium)}")
                st.markdown("**Recommendation**: Include these features. They provide moderate predictive value.")
            
            if weak:
                st.warning(f"⚠️ **Weak Features (0.02 ≤ IV < 0.1)**: {', '.join(weak)}")
                st.markdown("**Recommendation**: Consider excluding unless they have strong business justification or capture unique information.")
            
            if useless:
                st.error(f"❌ **Useless Features (IV < 0.02)**: {', '.join(useless)}")
                st.markdown("**Recommendation**: Exclude from model. They add no predictive value and may introduce noise.")
            
            # Feature engineering suggestions
            st.markdown("---")
            st.subheader("🔧 Feature Engineering Suggestions")
            
            st.markdown("""
            Based on the IV analysis, consider these feature engineering strategies:
            
            1. **WoE Encoding**: Transform top features using WoE values (especially categorical features)
            2. **Binning**: Create bins for continuous features with non-linear relationships
            3. **Interactions**: Create interaction terms between strong features
            4. **Polynomial Features**: Add squared/cubed terms for strong numeric features
            5. **Missing Indicators**: Create binary flags for features with missing values
            6. **Aggregations**: Create ratio features (e.g., debt-to-income, utilization rates)
            
            **Next Steps for Task 2:**
            - Export WoE transformations for strong features
            - Create interaction terms between top 3-5 features
            - Test polynomial transformations on numeric features with IV > 0.1
            - Handle missing values strategically (don't just drop!)
            """)
        
        else:
            st.error("Unable to calculate IV for any features.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>DataQuest 2026 - Credit Risk EDA Tool</strong></p>
    <p>Built with Streamlit | Powered by Plotly</p>
    <p>💡 Remember: Always validate findings with domain experts and comply with fair lending regulations</p>
</div>
""", unsafe_allow_html=True)