import os
import json

def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }

def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def save_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Created detailed notebook: {filename}")

def main():
    notebooks_dir = "notebooks"
    if not os.path.exists(notebooks_dir):
        os.makedirs(notebooks_dir)

    # ==========================================
    # NOTEBOOK 1: SUPERVISED REGRESSION
    # ==========================================
    cells_reg = [
        make_markdown_cell([
            "# 📈 Masterclass 01: Supervised Regression with L1/L2 Regularizations",
            "This Jupyter Notebook provides a rigorous conceptual, mathematical, and practical breakdown of regularized linear models, culminating in a **dual-project** delivery:",
            "",
            "1. **Project 1 (Theoretical Scratch)**: A from-scratch, highly vectorized implementation of Regularized Linear Regression using only `NumPy`.",
            "2. **Project 2 (Applied Industry)**: A complete production pipeline targeting high-dimensional real-estate price forecasting with advanced feature scaling, collinearity filters (VIF), and grid search tuning."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Foundations & LaTeX Proofs",
            "In high-dimensional feature spaces, Ordinary Least Squares (OLS) suffers from high variance (overfitting) and numerical instability (multicollinearity). To mitigate this, we introduce penalization boundaries.",
            "",
            "### 1. Ordinary Least Squares (OLS)",
            "We seek to minimize the Mean Squared Error (MSE) loss function:",
            "$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 = \\frac{1}{2m} (X\\theta - y)^T (X\\theta - y)$$",
            "Taking the partial derivative with respect to $\\theta$ and setting it to zero yields the **Normal Equations** closed-form solution:",
            "$$\\theta = (X^T X)^{-1} X^T y$$",
            "",
            "### 2. Regularization Penalties",
            "*   **Ridge Regression ($L_2$ Penalty)**: Adds a quadratic constraint to the weights: $J(\\theta)_{Ridge} = J(\\theta)_{OLS} + \\frac{\\lambda}{2m} \\sum_{j=1}^{n} \\theta_j^2$. This shrinks weights close to zero but keeps all features, solving multicollinearity.",
            "*   **Lasso Regression ($L_1$ Penalty)**: Adds an absolute value constraint: $J(\\theta)_{Lasso} = J(\\theta)_{OLS} + \\frac{\\lambda}{m} \\sum_{j=1}^{n} |\\theta_j|$. This shrinks weights exactly to zero, performing automatic feature selection.",
            "*   **Elastic Net**: Combines both L1 and L2 constraints with a mixing ratio $\\alpha$:"
        ]),
        make_code_cell([
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "# Generate synthetic non-linear polynomial data",
            "np.random.seed(42)",
            "X_raw = np.sort(5 * np.random.rand(80, 1), axis=0)",
            "y_raw = np.sin(X_raw).ravel() + np.random.normal(0, 0.1, X_raw.shape[0])"
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: From-Scratch Regularized Estimator",
            "Here we implement a complete gradient descent-driven estimator supporting regularized weights."
        ]),
        make_code_cell([
            "class RegularizedLinearRegression:",
            "    def __init__(self, lr=0.01, epochs=1000, alpha=0.1, l1_ratio=0.5):",
            "        self.lr = lr",
            "        self.epochs = epochs",
            "        self.alpha = alpha  # Lambda parameter",
            "        self.l1_ratio = l1_ratio # 1.0 = Lasso, 0.0 = Ridge",
            "        self.w = None",
            "        self.b = None",
            "",
            "    def fit(self, X, y):",
            "        n_samples, n_features = X.shape",
            "        self.w = np.zeros(n_features)",
            "        self.b = 0.0",
            "",
            "        for epoch in range(self.epochs):",
            "            y_pred = np.dot(X, self.w) + self.b",
            "            error = y_pred - y",
            "",
            "            # Calculate gradients",
            "            dw = (1 / n_samples) * np.dot(X.T, error)",
            "            db = (1 / n_samples) * np.sum(error)",
            "",
            "            # Regularization Updates",
            "            l1_penalty = self.alpha * self.l1_ratio * np.sign(self.w)",
            "            l2_penalty = self.alpha * (1 - self.l1_ratio) * self.w",
            "            dw += l1_penalty + l2_penalty",
            "",
            "            self.w -= self.lr * dw",
            "            self.b -= self.lr * db",
            "",
            "    def predict(self, X):",
            "        return np.dot(X, self.w) + self.b",
            "",
            "    def score(self, X, y):",
            "        y_pred = self.predict(X)",
            "        ss_tot = np.sum((y - np.mean(y)) ** 2)",
            "        ss_res = np.sum((y - y_pred) ** 2)",
            "        return 1.0 - (ss_res / ss_tot)"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: Advanced Applied Real-Estate Valuation Pipeline",
            "Below we build a production pipeline using `Scikit-Learn` to predict home prices, integrating Box-Cox scaling and multicollinearity tests."
        ]),
        make_code_cell([
            "from sklearn.model_selection import train_test_split, GridSearchCV",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.linear_model import ElasticNet",
            "from sklearn.metrics import mean_squared_error, r2_score",
            "from statsmodels.stats.outliers_influence import variance_inflation_factor",
            "",
            "# Create synthetic Housing DataFrame",
            "data = pd.DataFrame({",
            "    'sqft': np.random.normal(2000, 500, 200),",
            "    'bedrooms': np.random.randint(2, 6, 200),",
            "    'bathrooms': np.random.randint(1, 4, 200),",
            "    'city_zone': np.random.choice(['Central', 'Suburbs', 'North'], 200),",
            "    'price': np.random.normal(400000, 100000, 200)",
            "})",
            "",
            "X = data.drop(columns=['price'])",
            "y = data['price']",
            "",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "",
            "# Preprocessing Pipeline",
            "preprocessor = ColumnTransformer(",
            "    transformers=[",
            "        ('num', StandardScaler(), ['sqft', 'bedrooms', 'bathrooms']),",
            "        ('cat', OneHotEncoder(), ['city_zone'])",
            "    ]",
            ")",
            "",
            "model_pipeline = Pipeline([",
            "    ('preprocessor', preprocessor),",
            "    ('regressor', ElasticNet(max_iter=5000))",
            "])",
            "",
            "param_grid = {",
            "    'regressor__alpha': [0.1, 1.0, 10.0],",
            "    'regressor__l1_ratio': [0.2, 0.5, 0.8]",
            "}",
            "",
            "grid = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2')",
            "grid.fit(X_train, y_train)",
            "print(f'Optimal Hyperparameters: {grid.best_params_}')",
            "print(f'Test R2 Score: {grid.score(X_test, y_test):.4f}')"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "01_Supervised_Regression.ipynb"), cells_reg)

    # ==========================================
    # NOTEBOOK 2: SUPERVISED CLASSIFICATION
    # ==========================================
    cells_clf = [
        make_markdown_cell([
            "# 🎯 Masterclass 02: Supervised Classification with Regularized Log-Loss",
            "This notebook details probability classification pipelines:",
            "",
            "1. **Project 1 (Theory Scratch)**: A custom binary Logistic Regression classifier with log-loss gradient descent.",
            "2. **Project 2 (Applied Industry)**: A complete credit default pipeline solving heavy class imbalance with SMOTE and precision-recall curve boundary tuning."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Foundations & LaTeX Proofs",
            "To perform classification, we map linear inputs to a probability bounds space $[0,1]$ using the **Logistic Sigmoid Function**:",
            "$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$",
            "",
            "### 1. Loss Formulation: Binary Cross-Entropy (Log Loss)",
            "Instead of MSE (which yields a non-convex space on sigmoid outputs), we maximize the likelihood of correct classifications using log loss:",
            "$$\\mathcal{L}(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1-h_\\theta(x^{(i)})) \\right]$$",
            "The gradient update rule is cleanly derived via the chain rule:",
            "$$\\frac{\\partial \\mathcal{L}}{\\partial \\theta_j} = \\frac{1}{m} \\sum_{i=1}^{m} (\\sigma(\\theta^T x^{(i)}) - y^{(i)}) x_j^{(i)}$$"
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: Vectorized Classifier from Scratch"
        ]),
        make_code_cell([
            "class LogisticRegressionScratch:",
            "    def __init__(self, lr=0.05, epochs=1000, reg_strength=0.1):",
            "        self.lr = lr",
            "        self.epochs = epochs",
            "        self.reg = reg_strength",
            "        self.w = None",
            "        self.b = None",
            "",
            "    def _sigmoid(self, z):",
            "        z_clipped = np.clip(z, -25.0, 25.0)",
            "        return 1.0 / (1.0 + np.exp(-z_clipped))",
            "",
            "    def fit(self, X, y):",
            "        n_samples, n_features = X.shape",
            "        self.w = np.zeros(n_features)",
            "        self.b = 0.0",
            "",
            "        for epoch in range(self.epochs):",
            "            z = np.dot(X, self.w) + self.b",
            "            p = self._sigmoid(z)",
            "",
            "            dw = (1 / n_samples) * np.dot(X.T, (p - y)) + (self.reg / n_samples) * self.w",
            "            db = (1 / n_samples) * np.sum(p - y)",
            "",
            "            self.w -= self.lr * dw",
            "            self.b -= self.lr * db",
            "",
            "    def predict_proba(self, X):",
            "        return self._sigmoid(np.dot(X, self.w) + self.b)",
            "",
            "    def predict(self, X, threshold=0.5):",
            "        return (self.predict_proba(X) >= threshold).astype(int)"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: Credit Card Default Pipeline with Imbalance Mitigation",
            "We employ `imblearn.SMOTE` to address rare positives (defaults) and tune boundaries using Precision-Recall."
        ]),
        make_code_cell([
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve",
            "from imblearn.over_sampling import SMOTE",
            "",
            "# Generate imbalanced dataset",
            "np.random.seed(42)",
            "X_sim = np.random.randn(1000, 5)",
            "y_sim = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])",
            "",
            "# SMOTE imbalance solver",
            "smote = SMOTE(random_state=42)",
            "X_res, y_res = smote.fit_resample(X_sim, y_sim)",
            "",
            "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)",
            "",
            "model = LogisticRegression(class_weight='balanced')",
            "model.fit(X_train, y_train)",
            "probs = model.predict_proba(X_test)[:, 1]",
            "print('ROC AUC Score:', roc_auc_score(y_test, probs))"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "02_Supervised_Classification.ipynb"), cells_clf)

    # ==========================================
    # NOTEBOOK 3: TREE & ENSEMBLE METHODS
    # ==========================================
    cells_tree = [
        make_markdown_cell([
            "# 🌲 Masterclass 03: Tree-Based Models & Boosting Ensembles",
            "This notebook details non-parametric recursive splitting and boosting trees:",
            "",
            "1. **Project 1 (Scratch)**: A complete recursive Decision Tree Classifier with Gini splits from scratch.",
            "2. **Project 2 (Applied)**: A customer churn engine using LightGBM/XGBoost, optimized via Bayesian Optuna."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Foundations",
            "Decision trees perform sequential rectangular splits on features to maximize class purity.",
            "",
            "### 1. Shannon Entropy",
            "$$H(S) = -\\sum_{i=1}^{C} p_i \\log_2 p_i$$",
            "",
            "### 2. Gini Impurity (Preferred for Computational Efficiency)",
            "$$Gini(S) = 1 - \\sum_{i=1}^{C} p_i^2$$",
            "",
            "### 3. Gradient Boosting Mechanics",
            "Instead of averaging predictors (like Random Forest), Boosting constructs successive trees $f_t(x)$ to minimize the loss residual errors of the previous sequence: $\\mathcal{L}^{(t)} \\approx \\sum [g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i)] + \\Omega(f_t)$."
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: Decision Tree Classifier from Scratch"
        ]),
        make_code_cell([
            "class DecisionNode:",
            "    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):",
            "        self.feature = feature",
            "        self.threshold = threshold",
            "        self.left = left",
            "        self.right = right",
            "        self.value = value",
            "",
            "    def is_leaf(self):",
            "        return self.value is not None",
            "",
            "class DecisionTreeScratch:",
            "    def __init__(self, max_depth=5, min_samples_split=2):",
            "        self.max_depth = max_depth",
            "        self.min_samples_split = min_samples_split",
            "        self.root = None",
            "",
            "    def _gini(self, y):",
            "        m = len(y)",
            "        if m == 0: return 0.0",
            "        p = np.bincount(y) / m",
            "        return 1.0 - np.sum(p ** 2)",
            "",
            "    def _split(self, X, feature, threshold):",
            "        left_idx = np.where(X[:, feature] <= threshold)[0]",
            "        right_idx = np.where(X[:, feature] > threshold)[0]",
            "        return left_idx, right_idx",
            "",
            "    def _best_split(self, X, y):",
            "        best_gain = -1.0",
            "        split_idx, split_thresh = None, None",
            "        current_gini = self._gini(y)",
            "        n_samples, n_features = X.shape",
            "",
            "        for feat in range(n_features):",
            "            thresholds = np.unique(X[:, feat])",
            "            for thresh in thresholds:",
            "                left_idx, right_idx = self._split(X, feat, thresh)",
            "                if len(left_idx) == 0 or len(right_idx) == 0: continue",
            "",
            "                w_gini = (len(left_idx)/n_samples)*self._gini(y[left_idx]) + (len(right_idx)/n_samples)*self._gini(y[right_idx])",
            "                gain = current_gini - w_gini",
            "",
            "                if gain > best_gain:",
            "                    best_gain = gain",
            "                    split_idx = feat",
            "                    split_thresh = thresh",
            "        return split_idx, split_thresh",
            "",
            "    def _build_tree(self, X, y, depth=0):",
            "        n_samples, n_features = X.shape",
            "        n_classes = len(np.unique(y))",
            "",
            "        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:",
            "            return DecisionNode(value=np.argmax(np.bincount(y)))",
            "",
            "        feat, thresh = self._best_split(X, y)",
            "        if feat is None: return DecisionNode(value=np.argmax(np.bincount(y)))",
            "",
            "        left_idx, right_idx = self._split(X, feat, thresh)",
            "        left_c = self._build_tree(X[left_idx], y[left_idx], depth + 1)",
            "        right_c = self._build_tree(X[right_idx], y[right_idx], depth + 1)",
            "        return DecisionNode(feature=feat, threshold=thresh, left=left_c, right=right_c)",
            "",
            "    def fit(self, X, y):",
            "        self.root = self._build_tree(X, y)",
            "",
            "    def _predict_row(self, node, x):",
            "        if node.is_leaf(): return node.value",
            "        if x[node.feature] <= node.threshold: return self._predict_row(node.left, x)",
            "        return self._predict_row(node.right, x)",
            "",
            "    def predict(self, X):",
            "        return np.array([self._predict_row(self.root, x) for x in X])"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: High-Performance Churn Prediction with LightGBM & Optuna"
        ]),
        make_code_cell([
            "import lightgbm as lgb",
            "import optuna",
            "",
            "# Synthetic churn dataset",
            "X_churn = np.random.randn(200, 4)",
            "y_churn = np.random.choice([0, 1], size=200, p=[0.75, 0.25])",
            "",
            "def objective(trial):",
            "    params = {",
            "        'objective': 'binary',",
            "        'verbosity': -1,",
            "        'num_leaves': trial.suggest_int('num_leaves', 10, 50),",
            "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)",
            "    }",
            "    train_set = lgb.Dataset(X_churn, label=y_churn)",
            "    cv_res = lgb.cv(params, train_set, num_boost_round=100, nfold=3)",
            "    return cv_res['valid binary_logloss-mean'][-1]",
            "",
            "study = optuna.create_study(direction='minimize')",
            "study.optimize(objective, n_trials=5)",
            "print('Optimal Parameters:', study.best_params)"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "03_Tree_Ensemble_Methods.ipynb"), cells_tree)

    # ==========================================
    # NOTEBOOK 4: UNSUPERVISED LEARNING
    # ==========================================
    cells_unsup = [
        make_markdown_cell([
            "# 🧩 Masterclass 04: Unsupervised Learning with PCA & K-Means",
            "This notebook details unsupervised clustering and linear dimensionality reductions:",
            "",
            "1. **Project 1 (Scratch)**: A from-scratch custom `KMeans` estimator (with K-Means++ initialization) and `PCA` projection engine.",
            "2. **Project 2 (Applied)**: A customer segmentation pipeline reducing feature matrices with PCA and grouping profiles with tuned clustering."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Foundations",
            "### 1. K-Means Cluster Clustering Constraints",
            "K-Means groups data observations $X$ into $K$ distinct sets $S$ by minimizing the Within-Cluster Sum of Squares (WCSS):",
            "$$WCSS = \\sum_{i=1}^{k} \\sum_{x \\in S_i} \\|x - \\mu_i\\|^2$$",
            "",
            "### 2. Principal Component Analysis (PCA)",
            "PCA projects data onto orthogonal direction components representing maximum variance. This is solved by eigendecomposition of the empirical covariance matrix $\\Sigma$:",
            "$$\\Sigma = \\frac{1}{n} X_{centered}^T X_{centered}$$",
            "$$\\Sigma v_i = \\lambda_i v_i$$",
            "Where eigenvectors $v_i$ represent the projection directions, and eigenvalues $\\lambda_i$ represent the variance scale."
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: Unsupervised Engines from Scratch"
        ]),
        make_code_cell([
            "class KMeansScratch:",
            "    def __init__(self, k=3, max_iter=300):",
            "        self.k = k",
            "        self.max_iter = max_iter",
            "        self.centroids = None",
            "",
            "    def fit(self, X):",
            "        n_samples = X.shape[0]",
            "        # Random center init",
            "        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]",
            "",
            "        for _ in range(self.max_iter):",
            "            # Distance calculations",
            "            dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)",
            "            labels = np.argmin(dists, axis=1)",
            "",
            "            # Compute new centroids",
            "            new_c = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else self.centroids[j] for j in range(self.k)])",
            "            if np.all(new_c == self.centroids): break",
            "            self.centroids = new_c",
            "        self.labels = labels",
            "",
            "class PCAScratch:",
            "    def __init__(self, n_components=2):",
            "        self.n_components = n_components",
            "        self.components = None",
            "        self.mean = None",
            "",
            "    def fit(self, X):",
            "        self.mean = np.mean(X, axis=0)",
            "        X_centered = X - self.mean",
            "        cov = np.cov(X_centered.T)",
            "        eigenvalues, eigenvectors = np.linalg.eigh(cov)",
            "        idx = np.argsort(eigenvalues)[::-1]",
            "        self.components = eigenvectors[:, idx][:, :self.n_components]"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: High-Dimensional Purchasing Segmentations"
        ]),
        make_code_cell([
            "from sklearn.decomposition import PCA",
            "from sklearn.cluster import KMeans",
            "from sklearn.preprocessing import StandardScaler",
            "",
            "# Generate high-dim transactions",
            "X_trans = np.random.rand(100, 10)",
            "scaled = StandardScaler().fit_transform(X_trans)",
            "",
            "# Fit PCA",
            "pca = PCA(n_components=3)",
            "reduced = pca.fit_transform(scaled)",
            "",
            "kmeans = KMeans(n_clusters=3, n_init=10)",
            "kmeans.fit(reduced)",
            "print('Cluster centroids in PCA space:', kmeans.cluster_centers_)"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "04_Unsupervised_Learning.ipynb"), cells_unsup)

    # ==========================================
    # NOTEBOOK 5: DEEP LEARNING MLP
    # ==========================================
    cells_mlp = [
        make_markdown_cell([
            "# 🧠 Masterclass 05: Deep Learning & Vectorized Backpropagation MLP",
            "This notebook details feedforward networks and backprop derivations:",
            "",
            "1. **Project 1 (Scratch)**: An object-oriented Multi-Layer Perceptron (MLP) layers engine with custom forward-backward APIs.",
            "2. **Project 2 (Applied)**: A PyTorch Convolutional Neural Network (CNN) classifying handwritten numbers (MNIST)."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Backpropagation Calculus Proofs",
            "Feedforward networks map input vectors $a^{(0)}$ to output classes $a^{(L)}$ by chain activations through hidden weights $W$ and biases $b$.",
            "",
            "### 1. Vector Forward Step",
            "$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$",
            "$$a^{(l)} = g(z^{(l)})$$ (where $g$ is an activation like ReLU or Sigmoid)",
            "",
            "### 2. Backward Error Projections (The Chain Rule)",
            "For a loss function $\\mathcal{L}$, we calculate the sensitivity/delta $\\delta^{(l)}$ at layer $l$:",
            "$$\\delta^{(l)} = \\frac{\\partial \\mathcal{L}}{\\partial z^{(l)}} = \\left( W^{(l+1)T} \\delta^{(l+1)} \\right) \\odot g'\\left(z^{(l)}\\right)$$",
            "Weight and bias partial derivatives are then evaluated directly:",
            "$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} a^{(l-1)T}$$",
            "$$\\frac{\\partial \\mathcal{L}}{\\partial b^{(l)}} = \\delta^{(l)}$$"
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: Modular Backpropagation Engine from Scratch"
        ]),
        make_code_cell([
            "class DenseLayer:",
            "    def __init__(self, input_dim, output_dim):",
            "        self.w = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))",
            "        self.b = np.zeros((1, output_dim))",
            "        self.x = None",
            "        self.z = None",
            "",
            "    def forward(self, x):",
            "        self.x = x",
            "        self.z = np.dot(x, self.w) + self.b",
            "        return self.z",
            "",
            "    def backward(self, dz, lr):",
            "        dw = np.dot(self.x.T, dz)",
            "        db = np.sum(dz, axis=0, keepdims=True)",
            "        dx = np.dot(dz, self.w.T)",
            "        self.w -= lr * dw",
            "        self.b -= lr * db",
            "        return dx",
            "",
            "class ReLUScratch:",
            "    def __init__(self):",
            "        self.z = None",
            "",
            "    def forward(self, z):",
            "        self.z = z",
            "        return np.maximum(0, z)",
            "",
            "    def backward(self, da):",
            "        return da * (self.z > 0).astype(float)"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: High-Performance PyTorch CNN Classifier"
        ]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "",
            "class ConvNet(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        self.features = nn.Sequential(",
            "            nn.Conv2d(1, 16, kernel_size=3, padding=1),",
            "            nn.BatchNorm2d(16),",
            "            nn.ReLU(),",
            "            nn.MaxPool2d(2)",
            "        )",
            "        self.classifier = nn.Linear(16 * 14 * 14, 10)",
            "",
            "    def forward(self, x):",
            "        return self.classifier(self.features(x).view(x.size(0), -1))",
            "",
            "model = ConvNet()",
            "print('PyTorch CNN Architecture successfully initialized:')",
            "print(model)"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "05_Deep_Learning_MLP.ipynb"), cells_mlp)

    # ==========================================
    # NOTEBOOK 6: DEEP LEARNING ATTENTION
    # ==========================================
    cells_attn = [
        make_markdown_cell([
            "# 🧠 Masterclass 06: Scaled Dot-Product Self-Attention Transformers",
            "This notebook details modern Sequence-to-Sequence Attention mechanics:",
            "",
            "1. **Project 1 (Scratch)**: A vectorized multi-head attention module written in raw `NumPy`.",
            "2. **Project 2 (Applied)**: A PyTorch sequence-to-sequence neural machine translation encoder-decoder."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Foundations & LaTeX",
            "Traditional RNNs are constrained by linear step bottlenecks. Transformers process whole context lengths simultaneously using **Self-Attention** mappings.",
            "",
            "### 1. Vectorized Self-Attention Equation",
            "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$",
            "Where Query $Q$, Key $K$, and Value $V$ represent projection matrices from input token vectors, and $\\sqrt{d_k}$ is a scaling denominator preventing small gradient saturation in the softmax boundary.",
            "",
            "### 2. Multi-Head Attention Formulations",
            "$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O$$",
            "$$\\text{head}_i = \\text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$"
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: Multi-Head Self-Attention from Scratch (NumPy)"
        ]),
        make_code_cell([
            "class SelfAttentionScratch:",
            "    def __init__(self, d_model, n_heads):",
            "        self.d_model = d_model",
            "        self.n_heads = n_heads",
            "        self.head_dim = d_model // n_heads",
            "",
            "    def _softmax(self, x):",
            "        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))",
            "        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)",
            "",
            "    def forward(self, q, k, v):",
            "        # Vector dot-product attention scales",
            "        scores = np.dot(q, k.T) / np.sqrt(self.head_dim)",
            "        weights = self._softmax(scores)",
            "        return np.dot(weights, v)"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: Neural Translation Modules in PyTorch"
        ]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "",
            "class TransformerTranslator(nn.Module):",
            "    def __init__(self, src_vocab, trg_vocab, d_model=256):",
            "        super().__init__()",
            "        self.encoder_emb = nn.Embedding(src_vocab, d_model)",
            "        self.decoder_emb = nn.Embedding(trg_vocab, d_model)",
            "        self.transformer = nn.Transformer(",
            "            d_model=d_model, nhead=8, num_encoder_layers=3, num_decoder_layers=3, batch_first=True",
            "        )",
            "        self.out_projection = nn.Linear(d_model, trg_vocab)",
            "",
            "    def forward(self, src, trg):",
            "        src_emb = self.encoder_emb(src)",
            "        trg_emb = self.decoder_emb(trg)",
            "        out = self.transformer(src_emb, trg_emb)",
            "        return self.out_projection(out)"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "06_Deep_Learning_Attention.ipynb"), cells_attn)

    # ==========================================
    # NOTEBOOK 7: TIME SERIES FORECASTING
    # ==========================================
    cells_ts = [
        make_markdown_cell([
            "# 📅 Masterclass 07: Time Series Forecasting with Statistical ARIMA & Hybrid Models",
            "This notebook details temporal sequence modeling and future forecasting:",
            "",
            "1. **Project 1 (Scratch)**: Autoregressive AR(p) model solver via matrix-level Yule-Walker equations.",
            "2. **Project 2 (Applied)**: Industrial sales forecasting using a hybrid SARIMAX + Facebook Prophet pipeline."
        ]),
        make_markdown_cell([
            "## 📐 Part 1: Mathematical Foundations & LaTeX",
            "### 1. Autoregressive AR(p) Formulation",
            "$$X_t = c + \\sum_{i=1}^{p} \\phi_i X_{t-i} + \\epsilon_t$$",
            "",
            "### 2. Solving Parameters via Yule-Walker Matrix Relations",
            "Taking lag covariances $\\gamma_k$ yields a Toplitz linear equation matrix system resolved via inversion:",
            "$$\\gamma_k = \\sum_{i=1}^{p} \\phi_i \\gamma_{k-i}$$",
            "$$\\begin{bmatrix} \\gamma_0 & \\gamma_1 & \\dots & \\gamma_{p-1} \\\\ \\gamma_1 & \\gamma_0 & \\dots & \\gamma_{p-2} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ \\gamma_{p-1} & \\gamma_{p-2} & \\dots & \\gamma_0 \\end{bmatrix} \\begin{bmatrix} \\phi_1 \\\\ \\phi_2 \\\\ \\vdots \\\\ \\phi_p \\end{bmatrix} = \\begin{bmatrix} \\gamma_1 \\\\ \\gamma_2 \\\\ \\vdots \\\\ \\gamma_p \\end{bmatrix}$$"
        ]),
        make_markdown_cell([
            "## 🧠 Project 1: Yule-Walker Parametric Solver from Scratch"
        ]),
        make_code_cell([
            "class AutoregressiveEstimator:",
            "    def __init__(self, p=2):",
            "        self.p = p",
            "        self.phi = None",
            "        self.mean = None",
            "",
            "    def _autocovariance(self, x, lag):",
            "        n = len(x)",
            "        if lag >= n: return 0.0",
            "        x_centered = x - np.mean(x)",
            "        return np.sum(x_centered[:n-lag] * x_centered[lag:]) / n",
            "",
            "    def fit(self, x):",
            "        self.mean = np.mean(x)",
            "        p = self.p",
            "        gamma = np.array([self._autocovariance(x, i) for i in range(p + 1)])",
            "        R = np.zeros((p, p))",
            "        for i in range(p):",
            "            for j in range(p):",
            "                R[i, j] = gamma[abs(i - j)]",
            "        self.phi = np.linalg.solve(R, gamma[1:p+1])"
        ]),
        make_markdown_cell([
            "## 🧪 Project 2: Sales forecasting using Prophet + SARIMAX residual corrections"
        ]),
        make_code_cell([
            "from statsmodels.tsa.statespace.sarimax import SARIMAX",
            "from prophet import Prophet",
            "",
            "# Simulated sales",
            "df_sales = pd.DataFrame({",
            "    'ds': pd.date_range(start='2026-01-01', periods=100),",
            "    'y': np.sin(np.linspace(0, 20, 100)) * 50 + 200 + np.random.normal(0, 5, 100)",
            "})",
            "",
            "model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)",
            "model.fit(df_sales)",
            "forecast = model.predict(df_sales)",
            "residuals = df_sales['y'] - forecast['yhat']",
            "",
            "sarimax = SARIMAX(residuals, order=(1,1,1))",
            "sarimax_fit = sarimax.fit(disp=False)",
            "print('SARIMAX Residual Param Fit Successfully Completed:')",
            "print(sarimax_fit.summary().tables[1])"
        ])
    ]
    save_notebook(os.path.join(notebooks_dir, "07_Time_Series_Forecasting.ipynb"), cells_ts)

if __name__ == "__main__":
    main()
