# Causality-Uncertainty-Rail-Networks
Official implementation code for the paper 'From Prediction to Intervention,' a framework for causality and uncertainty in rail networks using GNNs, Causal ML, and Conformal Prediction.
This repository has been created to ensure the reproducibility of the analyses presented in our paper titled "From Prediction to Intervention: A Unified Framework for Causality and Uncertainty in Rail Networks using GNNs, Causal ML, and Conformal Prediction". In response to reviewer feedback received during the publication process, the Python implementation codes for all models and analyses used in the study, along with the datasets and relevant descriptions, are provided here.
Overview
All analysis programs used in this study use the file data.xlsx as the data source. The programs are numbered in accordance with the sequence specified in the paper. Each file contains a specific methodological analysis (e.g., Hold-Out, 5-Fold Cross-Validation (CV-5)) as described in the corresponding section of the article.
Dataset
The data required for all analyses are located in the data.xlsx file.
data.xlsx: This is the main dataset used in the analyses. All .py programs read this file directly to run. The column headers in this file are in Turkish.
data_ing.xlsx: This is a reference file containing the English equivalents of the Turkish column headers in data.xlsx. It has been included to facilitate understanding of the dataset.
Data Sources
The data used in this study were compiled from the following publicly available sources:
İBB Open Data Portal: RAIL SYSTEMS DAILY, MONTHLY, ANNUAL WAGON KILOMETER INFORMATION
Link: https://data.ibb.gov.tr/group/ulasim-hizmetleri?q=RAIL+SYSTEMS+DAILY%2C+MONTHLY%2C+ANNUAL+WAGON+KILOMETER+INFORMATION
Metro Istanbul: Lines Information
Link: https://www.metro.istanbul/Hatlarimiz/TumHatlarimiz

File Descriptions
The program files and their contents in this repository are as follows:
1.	1_cat_xg_hold_out_CV_5_bst.py
This script contains the analyses performed using CatBoost and XGBoost models. Both Hold-Out and 5-Fold Cross-Validation (CV-5) methods are implemented in this file.
2.	2_GNN_hold_ve_CV_.py
This script contains the analyses for the Graph Neural Networks (GNN) based model. The Hold-Out and 5-Fold Cross-Validation (CV-5) methods for this model are located in this script.
3.	3_causal_hold_out_CV_5_.py
This is the script for the Causal Machine Learning (Causal ML) analyses. Model results were generated using both Hold-Out and 5-Fold Cross-Validation (CV-5) methods.
4.	4_conformal_hold_out_CV_5_.py
This script contains the analyses based on the Conformal Prediction methodology. This file includes implementations for both Hold-Out and 5-Fold Cross-Validation (CV-5).
5.	5_causal_conformal_hold_out_CV_.py
This script contains the analyses for the hybrid framework that combines Causal Machine Learning and Conformal Prediction methods. Hold-Out and 5-Fold Cross-Validation (CV-5) methods are available in this script.
6.	6_gnn_causal_CV_hold_out_.py
This script contains the analysis codes for the hybrid framework combining GNN and Causal Machine Learning (Causal ML). The results for Hold-Out and 5-Fold Cross-Validation (CV-5) were obtained using this program.
7.	7_gnn_conformal_hold_out_CV_.py
This script contains the analyses for the hybrid framework combining GNN and Conformal Prediction. Implementations for Hold-Out and 5-Fold Cross-Validation (CV-5) are located in this file.
8.	8_GNN-enhanced frameworks_SHARP_ANALYSIS_.py
This script performs the SHAP (SHapley Additive exPlanations) analysis to determine model interpretability and feature importance for the GNN-enhanced frameworks.
Setup and Execution
You can follow the steps below to run the codes:
1.	Clone this repository or download the files to your computer.
2.	Ensure that Python and the necessary libraries (e.g., pandas, scikit-learn, catboost, xgboost, pytorch, pytorch-geometric, shap, etc.) are installed.
3.	Make sure that all .py scripts and the data.xlsx file are in the same directory.
4.	To run a specific analysis, execute the corresponding Python file from the command line (e.g., python 1_cat_xg_hold_out_CV_5_bst.py).
By providing these files, we aim to ensure that the findings presented in the paper are fully reproducible. Please feel free to contact us with any questions or feedback.

