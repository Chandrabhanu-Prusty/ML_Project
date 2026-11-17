# Syllabus mapping

| Module | Demonstrated concepts | Where to inspect |
| --- | --- | --- |
| 1 | Train/validation/test isolation, stratified CV, discrimination and operating-point metrics, overfitting | `src/train.py`, `src/evaluate.py`, Model Comparison and Model Insights |
| 2 | Numerical feature vectors, standardized Euclidean distances, covariance eigenvalues and SVD | Scaled/unscaled KNN trials, `src/visualization.py`, notebook 05 |
| 3 | Logistic decision boundary, L1/L2, C, kernel margins, k-nearest neighbors | `src/model_selection.py`, logistic/SVM/KNN CV tables, notebook 03 |
| 4 | MLP hidden layers, ReLU, backpropagation and Adam, early stopping | MLP grid and training loss, Model Insights, project report |
| 5 | Explained variance, PCA projection and t-SNE local neighborhoods | Dimensionality Reduction, notebook 05 |
| 6 | Seeded automation, versioned artifacts, Git ignore rules, experiment logs, persistence, serving contract | CLI, run histories, optional MLflow, FastAPI, pytest |
