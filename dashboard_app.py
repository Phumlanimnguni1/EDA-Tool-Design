import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score

# Optional: Make the dashboard use the full width of the screen
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Business Value Dashboard")
st.markdown("---")

# ==========================================
# 1. SIDEBAR: DATA UPLOADER
# ==========================================
st.sidebar.header("📂 Data Input")
st.sidebar.markdown("Upload your scored dataset to populate the dashboard.")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Stop execution until a file is uploaded
if uploaded_file is None:
    st.info("👋 Please upload a CSV dataset in the sidebar to get started.")
    st.markdown("""
    **Your dataset must include the following columns:**
    * `default_flag`: The actual historical outcome (0 = Good Loan, 1 = Default)
    * `pred_prob`: The model's predicted probability of default (values between 0.0 and 1.0)
    """)
    st.stop() 

# Read the uploaded file
try:
    results_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

# Validate that required columns exist
required_cols = ['default_flag', 'pred_prob']
missing_cols = [col for col in required_cols if col not in results_df.columns]

if missing_cols:
    st.error(f"❌ Missing required columns: **{', '.join(missing_cols)}**")
    st.warning("Please ensure your CSV exactly matches the required column names.")
    st.dataframe(results_df.head()) # Show what was actually uploaded to help debug
    st.stop()

# If we get here, data is loaded and valid!
st.sidebar.success("Dataset loaded successfully!")
st.sidebar.metric("Total Records", f"{len(results_df):,}")


# ==========================================
# 2. INTERACTIVE POLICY SLIDER
# ==========================================
st.header("1. Set Your Approval Policy")
threshold = st.slider(
    "Approval Threshold (Probability of Default Cut-off)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="Loans with predicted default probability below this threshold will be APPROVED"
)

st.markdown(f"**Policy Rule:** Approve if `pred_prob < {threshold:.0%}` | Reject if `pred_prob ≥ {threshold:.0%}`")
st.markdown("---")

# ==========================================
# 3. VOLUME VS. RISK TRADE-OFF
# ==========================================
st.header("2. Volume vs. Risk Trade-off")

# Calculate metrics for current threshold
approved_mask = results_df['pred_prob'] < threshold
approval_rate = approved_mask.mean()
approved_loans = results_df[approved_mask]
portfolio_risk = approved_loans['default_flag'].mean() if len(approved_loans) > 0 else 0

# Display dynamic metrics
col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Approval Rate",
        f"{approval_rate:.1%}",
        help="Percentage of total applicants approved"
    )
with col2:
    st.metric(
        "Portfolio Risk (Expected Default Rate)",
        f"{portfolio_risk:.1%}",
        help="Expected default rate among approved loans"
    )

# Generate trade-off chart across all thresholds
thresholds_range = np.linspace(0.05, 0.95, 91)
approval_volumes = []
portfolio_risks = []

for t in thresholds_range:
    approved_at_t = results_df['pred_prob'] < t
    approval_volumes.append(approved_at_t.mean() * 100)
    
    if approved_at_t.sum() > 0:
        portfolio_risks.append(results_df[approved_at_t]['default_flag'].mean() * 100)
    else:
        portfolio_risks.append(0)

# Create dual-axis Plotly chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=thresholds_range,
        y=approval_volumes,
        name="Approval Volume %",
        line=dict(color='#1f77b4', width=3),
        hovertemplate='Threshold: %{x:.0%}<br>Approval Volume: %{y:.1f}%<extra></extra>'
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=thresholds_range,
        y=portfolio_risks,
        name="Portfolio Risk %",
        line=dict(color='#d62728', width=3),
        hovertemplate='Threshold: %{x:.0%}<br>Portfolio Risk: %{y:.1f}%<extra></extra>'
    ),
    secondary_y=True
)

# Add vertical line for current threshold
fig.add_vline(
    x=threshold,
    line_dash="dash",
    line_color="green",
    line_width=2,
    annotation_text=f"Selected: {threshold:.0%}",
    annotation_position="top"
)

fig.update_xaxes(title_text="Approval Threshold (Probability of Default)", tickformat=".0%")
fig.update_yaxes(title_text="<b>Approval Volume %</b>", secondary_y=False, title_font=dict(color='#1f77b4'))
fig.update_yaxes(title_text="<b>Expected Portfolio Risk %</b>", secondary_y=True, title_font=dict(color='#d62728'))
fig.update_layout(
    title="Volume vs. Risk Trade-off Across All Thresholds",
    hovermode='x unified',
    height=500,
    showlegend=True,
    legend=dict(x=0.5, y=1.15, orientation='h', xanchor='center')
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# ==========================================
# 4. BUSINESS TRANSLATION OF ML METRICS
# ==========================================
st.header("3. Business Translation of ML Metrics")

# Calculate ML metrics at current threshold
y_true = results_df['default_flag']
y_pred_reject = (results_df['pred_prob'] >= threshold).astype(int)

precision = precision_score(y_true, y_pred_reject, zero_division=0)
recall = recall_score(y_true, y_pred_reject, zero_division=0)

# Display metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Precision", f"{precision:.1%}")
with col2:
    st.metric("Recall", f"{recall:.1%}")

st.markdown("")

# Business meanings
st.subheader("📊 What Do These Metrics Mean for Business?")

st.markdown(f"""
**Precision ({precision:.1%}):** Of all the loans we rejected, **{precision:.1%}** were actually going to default.  
✅ High precision = fewer false alarms and less missed revenue from rejecting good applicants.
""")

st.markdown(f"""
**Recall ({recall:.1%}):** Of all the actual bad loans in the population, **{recall:.1%}** were successfully rejected by our policy.  
✅ High recall = catching bad debt before it happens and protecting the portfolio.
""")

# Executive Summary
st.markdown("---")
st.success(f"""
💼 **Executive Summary** At a **{threshold:.0%}** approval threshold, your policy will:  
• Approve **{approval_rate:.1%}** of all applicants  
• Face an expected portfolio default rate of **{portfolio_risk:.1%}** • Successfully catch **{recall:.1%}** of all bad loans  
• Have **{precision:.1%}** accuracy in rejections (avoiding false alarms)  
""")