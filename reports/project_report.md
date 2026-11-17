# Comparative Analysis of ML Models for Credit Card Fraud Detection

## Abstract

This project compares seven supervised approaches to rare credit card fraud detection on anonymized transaction data from ULB and Worldline. It uses a stratified holdout, training-only tuning, and average precision for model selection. Model pipelines, thresholds, figures and run metadata are published together for reproducibility. The experiment also investigates regularization, training and validation score gaps, and two-dimensional feature maps. See [measured results](results.md) for the observed comparison and interpretation.

## Introduction and problem statement

Investigating a legitimate transaction has a cost, and missing fraud has a different cost. With fraud much rarer than legitimate activity, a system that classifies every transaction as legitimate can show high accuracy while detecting zero fraud. This study asks which model ranks rare positives well and how its validation-selected operating point changes the investigation workload.

## Objectives

- Profile the dataset and validate its feature contract.
- Evaluate seven classifiers and compare fold-local imbalance treatments.
- Study the effects of model complexity, feature scaling and regularization.
- Preserve reproducible scoring behavior in an API and an interactive dashboard.

## Background and dataset

The dataset describes anonymized European card transactions collected during a short period in September 2013. It provides elapsed seconds, amount, 28 principal components and confirmed class labels. The source PCA suppresses commercial feature names. Any association involving a component describes the published feature space. Class imbalance particularly affects precision because small false-positive rates can still yield many alerts. The [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) describes the collaboration and published attribution.

## Proposed methodology and architecture

The CSV loader rejects malformed columns, infinite values, nonnumeric fields and invalid labels. EDA describes missingness, duplicates, correlations, amount and time distributions, class-specific statistics and IQR flags. Modeling collapses exact duplicates and excludes identical feature vectors with conflicting labels. This avoids exact observed duplicates crossing partitions. EDA counts refer to the original data. The experiment retains outliers and handles partial missing features with training-derived median values.

An 80/20 stratified split uses seed 42. The training pool is divided again, with 20% reserved for model selection and operating thresholds. The remaining fitting partition is used for stratified grid search and refitting each winning configuration. Imputation, scaling and SMOTE are fitted inside the fold pipeline. The logistic imbalance conditions compare the observed distribution, inverse-frequency class weighting and SMOTE. For this dataset SMOTE targets a 0.05 post-resampling minority/majority ratio. Validation and holdout data retain the natural class prevalence after duplicate handling.

The orchestration CLI creates a versioned artifact set. FastAPI and Streamlit load the registered pipeline and use the same feature validation and scoring functions. The scoring path applies the fitted preprocessing, obtains a continuous score and flags records at or above the stored threshold.

## Algorithms

| Model | Mechanism | Experimental comparison |
| --- | --- | --- |
| Logistic regression | Linear log-odds with sigmoid link | L1 sparsity, L2 shrinkage, C, weighting, SMOTE |
| Decision tree | Recursive impurity-reducing feature splits | Maximum depth, minimum split size, class weighting |
| Random forest | Bagged trees with random feature selection | Depth, minimum leaf size, bootstrapped class weights |
| SVM | Maximum-margin classification | Linear and RBF kernels, C and gamma; signed decision scores |
| KNN | Local class vote based on distances | k and scaling; feature magnitude affects Euclidean neighborhoods |
| MLP | ReLU hidden layers and sigmoid binary output | Hidden-layer sizes and L2 alpha, Adam, early stopping |
| XGBoost | Sequential additive trees fit to loss derivatives | Depth, learning rate and positive gradient weighting |

The logistic solver uses LIBLINEAR. In a gradient-based model, derivatives of loss determine the update direction. The MLP uses backpropagation to apply the chain rule from output loss to hidden-layer weights, then Adam adjusts updates using running gradient moments. Its built-in early stopping monitors accuracy on an internal validation partition. This criterion is insensitive to rare positives, so the outer CV still compares candidates on AP. XGBoost adds trees sequentially using loss gradients and curvature. Random forest aggregates trees fitted with sampling and feature randomness.

## Experimental setup

Five stratified folds select each full-profile configuration by mean AP. Standard deviation gives the observed spread across folds. SVM and KNN use stratified fitting subsets because kernel and neighbor computations scale poorly. The common training-side validation set supplies the final model ranking by AP and one threshold per model by maximum F1. The largest threshold breaks F1 ties. Models are frozen before scoring the shared holdout. The training time includes CV search and final fit. Inference time measures one full holdout scoring pass and includes preprocessing.

AP is the recall-increment-weighted mean precision over score thresholds. ROC AUC describes ranking relative to negatives, and its false-positive rate can mask heavy alert volume. The confusion matrix captures one operating point. Precision, recall, F1, accuracy, false-positive rate and false-negative rate derive from that point. Average precision and ROC AUC use continuous scores.

## Bias–variance and regularization

Learning curves use stratified subsamples within a diagnostic training pool and the first three folds of its stratified partition. They compare in-sample and validation AP at three sample sizes. Persistent separation suggests sensitivity to the training sample. Low curves together may reflect limited model capacity or poor signal. These are diagnostics of the selected configurations, and the CV trial files provide a separate look at regularization and complexity changes. The test partition has no role in these diagnostics.

L1 in logistic regression promotes exactly zero coefficients. L2 penalizes squared weights and shrinks coefficients smoothly. Lower C strengthens either penalty. Tree depth and minimum split or leaf size constrain possible partitions. Bagging aims to reduce variance by averaging partially decorrelated trees. Early stopping limits how long MLP weights adapt to the training data.

## Dimensionality reduction

The PCA analysis fits its median imputer, scaler and singular value decomposition on the fitting partition. Right singular vectors define projection directions. Squared singular values divided by the sample degrees of freedom correspond to sample covariance eigenvalues. Explained variance fractions quantify the retained variation. The plotted PCA and t-SNE sample mixes random legitimate rows with enriched fraud rows for visibility. It does not represent population prevalence. t-SNE emphasizes local neighborhoods, and global distances and separation in the plot do not quantify predictive accuracy.

## Results and comparative analysis

The [measured results](results.md) record dataset counts, evaluation metrics, operating thresholds, learning diagnostics and plot captions from the executed run. Detailed configuration and trial-level scores are written to the run directory, and the dashboard exposes the comparisons.

## Discussion and limitations

The thresholds optimize F1 under one validation sample and may move when prevalence or costs change. Majority class alerts and missed fraud must be considered alongside AP. Capped SVM and KNN runs compare complete practical configurations, including differences in available training rows. Their results do not isolate algorithm effects at equal training size. Correlated anomalies can survive exact deduplication, and the dataset lacks card identifiers for grouped validation. Random splitting offers no guarantee for future transactions. Impurity importance and model coefficients describe associations rather than causal effects.

## Conclusion and future scope

The workflow produces reproducible comparisons, validation-selected operating points and deployable scoring artifacts for an academic demonstration. Future work could test chronological splits, probability calibration, cost-sensitive threshold optimization, bootstrap intervals and drift detection.

## References

1. ULB Machine Learning Group and Worldline. [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Carcillo, F., Dal Pozzolo, A., Le Borgne, Y.-A., Caelen, O., Mazzer, Y., and Bontempi, G. (2017). [SCARFF: a Scalable Framework for Streaming Credit Card Fraud Detection with Spark](https://arxiv.org/abs/1709.08920).
