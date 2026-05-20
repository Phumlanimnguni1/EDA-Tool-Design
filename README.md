# 📊 Credit Risk EDA Tool & Business Dashboard - DataQuest 2026

*Note: This repository contains the interactive Streamlit Data Applications (EDA Tool & Business Dashboard) for the FNB DataQuest 2026 project. 
The Data Engineering (ETL) and Logistic Regression Modelling pipelines are hosted in our main sister repository: **https://github.com/Phumlanimnguni1/credit-risk-logistic-modeling-interpretable-ml.**

An interactive Exploratory Data Analysis (EDA) and Decision Support tool built with Streamlit for analysing credit risk data, identifying predictive non-linear patterns, and simulating business lending policies. 

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Plotly](https://img.shields.io/badge/plotly-interactive-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project directly answers **Task 1** and **Bonus Task 3** of the DataQuest 2026 competition, which focuses on building interpretable credit models for retail lending. 

Because the business mandates a linear Logistic Regression model for regulatory transparency, a static data notebook is insufficient. This dynamic tool was engineered to help data scientists manually hunt for complex, non-linear risk clusters and translate machine learning probabilities into actionable business value.

---

## 🛠️ Key Features & Application Tabs
### 1. 🔍 Data Quality Report
c:\Users\prosp\OneDrive\Desktop\DE Tools\github projects\Quality report EDA.png
c:\Users\prosp\OneDrive\Desktop\DE Tools\github projects\EDA report Q.png
Automated data integrity auditing for the `loan_book` dataset:
- Missing value analysis with visualisations.
- Duplicate detection and Outlier identification (IQR method).
- Text formatting inconsistencies (e.g., standardising `home_ownership` casing).
- Target variable class imbalance detection.

### 2. 📊 Univariate Explorer (WoE / IV)
Deep-dive into individual features using industry-standard credit risk mathematics:
- Automated Weight of Evidence (WoE) calculation and visualisation to ensure monotonic trends.
- Information Value (IV) scoring to instantly identify predictive power.
- Default rate analysis by quantiles/bins.

### 3. 🔗 Bivariate Explorer (The Interaction Hunter)
c:\Users\prosp\OneDrive\Desktop\DE Tools\github projects\Debt Servicing Stress Index.png
The core engine for discovering hidden subgroup risks:
- Interactive Plotly scatter plots and density heatmaps overlaid with the target `default_flag`.
- Correlation matrices and interaction effect identification to guide mathematical feature engineering.

### 4. 🏆 Feature Ranking
Automated feature importance reporting:
- Information Value (IV) based ranking of all features (from "Useless" to "Strong Predictive Power").
- Feature selection recommendations and downloadable ranking reports to guide the Databricks ETL pipeline.

### 5. 💼 Business Value Dashboard 
c:\Users\prosp\OneDrive\Desktop\DE Tools\github projects\dashboard1EDA.png
c:\Users\prosp\OneDrive\Desktop\DE Tools\github projects\dashboard2EDA.png
c:\Users\prosp\OneDrive\Desktop\DE Tools\github projects\dashboard3EDA.png
A dynamic decision-support tool turning model outputs into lending strategies:
- **Interactive Policy Slider:** Allows risk managers to adjust the Probability of Default approval threshold.
- **Volume vs. Risk Trade-off:** A dual-axis visual simulation showing how tightening the policy impacts loan approval volume vs. expected portfolio risk.
- **Business Translation of ML Metrics:** 
Dynamically explains Precision and Recall in business terms (e.g. translating a 60.3% Precision into exactly how many false alarms the bank is avoiding).

---

## 💡 Key Discoveries (Driven by this Tool)

By utilizing the **Bivariate Explorer's** density heatmaps, this tool successfully uncovered several non-linear risk clusters that a standard logistic regression model would have missed. These visual discoveries directly drove the creation of our most powerful engineered features:

1. **Debt Servicing Stress Index:** Heatmaps revealed a dangerous hotspot where high `dti_ratio` and high `interest_rate` intersect, leading to the creation of a multiplied penalty feature.
2. **Loan-to-Income Ratio:** Scatterplots showed an 'L-shape' with near-zero correlation, proving raw loan amounts don't drive defaults, but *disproportionate* borrowing does.
3. **Delinquency Concentration:** By comparing historical delinquencies against the age of the credit profile, the tool helped us mathematically separate "chronic defaulters" from applicants with isolated historical rough patches.
4. **High Earners with Low Stability:** Identified isolated default anomalies where high salaries masked the risk of extremely short employment histories.

---

## 🔗 Sister Repository: ETL & Modelling Pipeline

To see how the insights generated from this Streamlit application were deployed into a production-grade machine learning pipeline, please visit our main repository: **https://github.com/Phumlanimnguni1/credit-risk-logistic-modeling-interpretable-ml**. 

The main repository contains:
- Databricks Delta Lake architecture (Bronze, Silver, Gold layers).
- Weight of Evidence (WoE) transformation pipelines (strictly fitted on training data to prevent leakage).
