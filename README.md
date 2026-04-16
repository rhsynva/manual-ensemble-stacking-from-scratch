# manual-ensemble-stacking-from-scratch
In this notebook, I have implemented the concept of Ensemble Stacking for regression from scratch. Instead of relying on pre-built wrapper functions like scikit-learn's StackingRegressor, I manually constructed the out-of-fold (OOF) prediction logic and built the meta-learner using linear algebra. Here is a breakdown of the workflow:

1.Data Loading and Exploration: I loaded the Boston Housing dataset and performed basic exploratory data analysis, including checking for missing values, generating summary statistics, and visualizing feature correlations using a Seaborn heatmap.

2.Data Preprocessing: I split the data into 80% training and 20% testing sets. Because models like KNN and SVR are sensitive to the scale of the data, I applied standard scaling (StandardScaler) to normalize the features.

3.Base Learners: I selected three diverse base models to capture different patterns in the data:
 1.K-Nearest Neighbors Regressor (n_neighbors=5)
 2.Support Vector Regressor (kernel='rbf')
 3.Decision Tree Regressor (max_depth=5)

4. Manual K-Fold Cross-Validation (The Core Stacking Logic): Instead of training the base models on the entire training set (which causes severe data leakage when training the meta-model), I manually implemented a 6-fold cross-validation loop. For each fold and each model, I trained on 5 folds and predicted on the holdout fold to build the oof_train matrix. Simultaneously, I generated predictions on the unseen test set and averaged them across the 6 folds to create the oof_test matrix.

5.Manual Meta-Learner (Normal Equation): Instead of using a built-in LinearRegression class for the final stage, I implemented the meta-learner mathematically. I added a bias column (intercept) to the OOF matrices and used the Normal Equation ($\beta = (X^T X)^{-1} X^T y$) via numpy.linalg.pinv to calculate the optimal weights for combining the base models.

6.Evaluation: I applied the calculated weights to the meta-test set to generate final predictions, achieving a highly competitive Mean Squared Error (MSE) of approximately 6.97.


This repository contains my implementation of an Ensemble Stacking Regressor built from scratch using Python and NumPy. Rather than relying on high-level library wrappers, I manually programmed the out-of-fold (OOF) cross-validation mechanics and the matrix operations for the meta-learner to deeply understand the mathematics and architecture behind stacked generalization. Stacking is an advanced ensemble technique that combines multiple predictive models via a meta-model. The challenge in stacking is avoiding data leakage; if the meta-model is trained on the same data used to train the base models, it will heavily overfit. My code demonstrates how to properly circumvent this using a manual K-Fold prediction strategy, culminating in a custom linear regression meta-learner built purely with linear algebra.

Technologies Used:
1.Python 3

2.NumPy for matrix algebra and numerical computing

3.Pandas for data manipulation

4.Scikit-learn for base model algorithms and preprocessing

5.Matplotlib and Seaborn for data visualization.
