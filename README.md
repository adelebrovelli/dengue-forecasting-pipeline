# Dengue Forecasting and Outbreak Prediction

Undergraduate thesis developed at UNICAP comparing SARIMA, Random Forest, and Bidirectional Multitask LSTM models for weekly dengue forecasting and outbreak detection using SINAN public health data from Jaboatão dos Guararapes (2018–2025).

The code demonstrates all stages of the project, including data preprocessing, feature engineering, model development, training, hyperparameter tuning, evaluation, and comparative analysis of statistical, machine learning, and deep learning approaches.

## Results

| Model          | Task                        | MAPE   | MedAPE | Accuracy | Precision | Recall | F1-score | AUC-ROC |
| -------------- | --------------------------- | ------ | ------ | -------- | --------- | ------ | -------- | ------- |
| SARIMA         | Regression                  | 96.37% | 66.48% | —        | —         | —      | —        | —       |
| Random Forest  | Classification              | —      | —      | 0.944    | 0.969     | 0.914  | 0.941    | 0.945   |
| Multitask LSTM | Regression + Classification | 64.92% | 49.04% | 0.972    | 0.961     | 0.980  | 0.970    | 0.972   |

The results indicate that the Bidirectional Multitask LSTM achieved the best overall performance among the evaluated approaches, outperforming the single-task statistical and machine learning models in both forecasting and outbreak detection tasks. The findings reinforce the potential of multitask deep learning architectures for epidemiological surveillance in municipal public health scenarios with limited historical data.

