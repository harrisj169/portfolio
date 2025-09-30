---
title: "Disease Prediction"
author: "James Harris"
date: "2025-09-30"
output:
  pdf_document: default
  html_document: default
---

# Symptom Smart: Disease Prediction Using Machine Learning

## Project Overview
*Symptom Smart* is my Master’s in Data Analytics capstone project. It explores how predictive analytics and machine learning can assist healthcare professionals by predicting diseases based on symptoms.  
The goal is not to replace clinical judgment, but to provide a **decision-support tool** that reduces diagnostic burden and helps surface likely conditions faster.

This project focuses on a rule based model and two tree-based classifiers:
- **Ripper Rule Learner**
- **Decision Tree**
- **Random Forest**

Both models were trained and tested on a dataset of **131 symptoms** mapped to **41 diseases**.

---

## Dataset
- **Source:** Public dataset from Kaggle (synthetically generated)  
- **Size:** 3,690 samples  
- **Features:** 131 binary symptom indicators  
- **Target Variable:** Disease (41 classes)  
- **Distribution:** Equal representation of all diseases (rare in real-world data)

---

## Methodology
1. **Data Preparation**  
   - Cleaned and standardized data using Excel and R  
   - Converted categorical features into binary (1 = present, 0 = absent)  
   - Partitioned data 75/25 for training/testing using stratified sampling  

2. **Model Training** 
   - Ripper Rule Learner: 68 rule transparent rules
   - Decision Tree: 74 nodes, interpretable baseline model
   - Random Forest: 500 trees, 11 variables per split, ensemble approach
   

3. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall (Sensitivity)  
   - Specificity  
   - F1 Score  

---

## Results
- **Ripper Rule Learner:** >99% accuracy, $\geq 90\%$ across other metrics  
- **Decision Tree:** >99% accuracy, $\geq 93\%$ across other metrics  
- **Random Forest:** 100% accuracy, $\geq 99\%$ across all metrics  
- **Feature Importance:** Symptoms such as *fever*, *cough*, *fatigue*, and *abnormal menstruation* were most discriminative  
- Random Forest proved more robust and stable, while Decision Tree was easier to interpret  

---

## Key Insights
- **Strengths:**  
  - High predictive accuracy on balanced dataset 
  - Strong interpretability (Decision Tree & Ripper Rule Learner)  
  - Reduced overfitting and variance (Random Forest)  

- **Limitations:**  
  - Dataset was clean and balanced (unlike real-world data)  
  - Lacked severity, duration, demographics, or patient history  
  - Model assumed a single disease per patient (no comorbidities)  
  - Rare diseases may be overlooked in real-world application  

---

## Future Work
- Integrate **real patient data** from Electronic Health Records (EHRs)  
- Support **multi-label classification** for comorbidities  
- Apply **feature weighting** based on severity and clinical relevance  
- Address **imbalanced class distributions** with oversampling or cost-sensitive learning  
- Regionalize data to track **local and global disease trends**  
- Develop a prototype **web or clinical app** for real-time use  

---

## Project Materials
- **Presentation Slides**: `Symptom_Smart.pdf`  
- **Final Report**: `Symptom_Smart_Report.pdf`  
- **README.md**: This file  

---

## Author
**James Harris**  
Master’s in Data Analytics Candidate – Class of 2025  
Capstone Project  

---

