# 📊 Credit Risk EDA Tool - DataQuest 2026

An interactive Exploratory Data Analysis (EDA) tool built with Streamlit for analyzing credit risk data and identifying predictive patterns in loan applications. This tool is designed to help data scientists and credit analysts understand their data, evaluate feature importance, and prepare for building interpretable credit models.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project is **Task 1** of the DataQuest 2026 competition, which focuses on building interpretable credit models for retail lending. The tool provides comprehensive data exploration capabilities with a focus on credit-specific metrics like **Weight of Evidence (WoE)** and **Information Value (IV)**.

### Key Features

- 📚 **Research & Concepts**: Educational content on credit modeling fundamentals
  - GLMs vs Non-Linear Models comparison
  - Interpretability vs Complexity trade-offs
  - Credit modeling metrics (WoE, IV, AUC, Gini)
  - Regulatory considerations and prohibited features

- 🔍 **Data Quality Report**: Comprehensive data integrity checks
  - Missing value analysis with visualizations
  - Duplicate detection
  - Outlier identification (IQR method)
  - Text formatting inconsistencies
  - Correlation analysis
  - Summary statistics

- 📊 **Univariate Explorer**: Deep-dive into individual features
  - Feature distribution by target variable
  - Weight of Evidence (WoE) calculation and visualization
  - Information Value (IV) scoring
  - Default rate analysis by bins
  - Downloadable WoE tables

- 🔗 **Bivariate Explorer**: Relationship analysis between features
  - Scatter plots and density heatmaps (numeric vs numeric)
  - Box plots and violin plots (numeric vs categorical)
  - Grouped bar charts (categorical vs categorical)
  - Correlation matrices
  - Interaction effect identification

- 🏆 **Feature Ranking**: Automated feature importance scoring
  - IV-based ranking of all features
  - Color-coded predictive power categories
  - Feature selection recommendations
  - Downloadable ranking reports
  - Feature engineering suggestions



