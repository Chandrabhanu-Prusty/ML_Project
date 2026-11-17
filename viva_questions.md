# Viva questions

1. **Why is fraud detection a rare-event problem?** Fraud cases form a very small fraction of the observed transactions.
2. **Why can accuracy be misleading?** Predicting every row as legitimate achieves high accuracy and zero fraud recall.
3. **What is precision?** The fraction of flagged transactions that are fraudulent.
4. **What is recall?** The fraction of actual fraud transactions the model flags.
5. **What is F1?** The harmonic mean of precision and recall.
6. **What does average precision measure?** Precision averaged over increases in recall as the score threshold moves.
7. **How does AP differ from trapezoidal PR AUC?** AP weights observed precision by recall increments, while trapezoidal area linearly interpolates between points.
8. **How do false positives affect operations?** They consume review capacity and can inconvenience legitimate customers.
9. **What is a false negative?** An actual fraud that stays below the operating threshold.
10. **What does ROC AUC describe?** How well scores rank positives above negatives across operating points.
11. **Why is split stratification useful?** It keeps the class proportions approximately comparable across partitions.
12. **Why remove exact duplicates before splitting?** Their presence in multiple partitions can inflate estimates by rewarding memorization.
13. **Why retain IQR-flagged values?** Extreme transactions can be genuine observations or useful fraud signals.
14. **How do you prevent imputation leakage?** Fit a median imputer within each training fold and apply the fitted statistics to validation.
15. **What is SMOTE?** Minority oversampling that interpolates between a minority point and a minority neighbor.
16. **Why must SMOTE run inside CV?** It must generate examples using only that fold's training records.
17. **What does class weighting do?** It changes the relative penalties for errors in each class.
18. **How is the operating threshold chosen?** Maximize F1 on a dedicated training-side validation partition.
19. **How is the final model selected?** Compare validation AP after each algorithm's hyperparameters are selected by CV.
20. **Why use continuous scores for ROC AUC and AP?** Curves require operating points across a range of score thresholds.
21. **How do L1 and L2 differ?** L1 penalizes absolute coefficients and can produce exact zeros. L2 penalizes squared coefficients.
22. **What happens as logistic C decreases?** The strength of regularization increases.
23. **Why can a decision tree overfit?** Deep partitions can memorize rare details that fail to recur in validation.
24. **How does a random forest reduce variance?** It averages varied trees built with randomized data and feature sampling.
25. **How does boosting differ from bagging?** Boosting adds models sequentially to improve the current objective. Bagging fits separate models to aggregate their predictions.
26. **What is an SVM margin?** The distance from the decision boundary to the closest support vectors in feature space.
27. **What does the RBF kernel add?** It supports nonlinear boundaries using distance-based similarities.
28. **Why use a subset for SVM and KNN?** Kernel fitting and brute-force distance scoring grow expensive as row counts increase.
29. **Why scale inputs for KNN?** Otherwise variables with larger numeric scales can dominate Euclidean distance.
30. **What does backpropagation calculate?** The derivatives of the loss with respect to network weights via the chain rule.
31. **What is early stopping?** Training stops when an internal validation metric stops improving for a specified number of epochs.
32. **What does a wide learning-curve gap suggest?** The model fits its training sample substantially better than unseen examples, consistent with overfitting.
33. **What do PCA axes represent?** Orthogonal directions ordered by how much training variance they explain.
34. **How do SVD and PCA relate?** The right singular vectors of the centered matrix are PCA directions.
35. **Why enrich fraud in the t-SNE plot?** A prevalence-matched sample could contain too few fraud points for visual inspection.
36. **How do you avoid training-serving skew?** Load and run the same serialized pipeline with the same canonical feature schema.
37. **Are fraud scores calibrated probabilities?** Estimator probability outputs can be miscalibrated, especially after weighting or oversampling. The SVM reports a decision margin.
38. **How could the next experiment assess drift?** Train on earlier transactions and measure performance on a later period, tracking score and label distributions.
