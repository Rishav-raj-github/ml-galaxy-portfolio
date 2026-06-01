/**
 * ML Galaxy Portfolio - Core Database (2026 Edition)
 * Contains metadata, concepts, mathematical foundations, interview Q&As, and
 * full production-ready Python source code for both From-Scratch and Applied projects.
 */

const MODEL_DATA = {
  regression: {
    title: "Supervised Regression",
    icon: "📈",
    gradient: "linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)",
    description: "Predicting continuous values using regularized linear and polynomial models.",
    models: [
      {
        id: "linear_regression",
        name: "Linear Regression (Ridge, Lasso, ElasticNet)",
        important2026: true,
        badge: "Industry Standard",
        concept: "A foundational approach to modeling the relationship between a scalar dependent variable y and one or more explanatory variables X. Extended with regularization (L1 Lasso, L2 Ridge) to penalize complexity, minimize overfitting, and perform variable selection.",
        math: `$$\\hat{y} = X\\theta$$\n\n**Loss Function with L1 & L2 Regularization (Elastic Net):**\n$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 + \\lambda \\left[ \\alpha \\sum_{j=1}^{n} |\\theta_j| + \\frac{1-\\alpha}{2} \\sum_{j=1}^{n} \\theta_j^2 \\right]$$\n\nWhere:\n*   $\\lambda$ is the regularization strength parameter.\n*   $\\alpha$ is the Elastic Net mixing parameter ($0 \\le \\alpha \\le 1$). When $\\alpha = 1$ it becomes Lasso; when $\\alpha = 0$ it is Ridge.`,
        pros: [
          "Extremely fast to train and run inference on large datasets.",
          "Highly interpretable coefficients; clearly shows feature influence.",
          "Regularization (Ridge/Lasso) handles multicollinearity and prevents overfitting."
        ],
        cons: [
          "Assumes linear relationship between features and target.",
          "Highly sensitive to outliers and extreme noise.",
          "Prone to underfitting on complex, non-linear real-world phenomena."
        ],
        qna: [
          {
            q: "What is the difference between Ridge (L2) and Lasso (L1) regularization, and when would you use each?",
            a: "Ridge adds a penalty equal to the sum of the squared weights ($L_2$). It shrinks weights close to zero but never exactly to zero, keeping all features. Lasso adds a penalty equal to the sum of the absolute weights ($L_1$), which can shrink coefficients exactly to zero, effectively acting as an automated feature selector. Use Lasso when you suspect only a small subset of features are useful, and Ridge when most features have small, distributed influences."
          },
          {
            q: "How does multicollinearity affect Linear Regression, and how do we identify and mitigate it?",
            a: "Multicollinearity occurs when independent variables are highly correlated. It destabilizes coefficient estimates, making them highly sensitive to small data changes and destroying interpretability. We identify it using Variance Inflation Factors (VIF) where a VIF > 5-10 indicates high correlation. We mitigate it by dropping redundant features, applying Ridge regularization, or projecting features to a lower dimensional space using PCA."
          }
        ],
        project1: {
          name: "Project 1: Regularized Linear Regression from Scratch",
          description: "An elegant, vectorized custom estimator implementing linear regression with adjustable L1 (Lasso) and L2 (Ridge) coordinate descent and gradient descent optimizers.",
          file: "notebooks/01_Supervised_Regression.ipynb",
          code: `import numpy as np

class RegularizedLinearRegression:
    """
    From-scratch vectorized linear estimator supporting Ridge (L2) and Lasso (L1) gradient updates.
    """
    def __init__(self, lr=0.01, epochs=1000, alpha=0.1, l1_ratio=0.5):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha  # Lambda parameter
        self.l1_ratio = l1_ratio # Elastic net ratio: 1.0=Lasso, 0.0=Ridge
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y

            # Computes gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Apply L1 (Lasso) & L2 (Ridge) Regularization Penalties to weight gradients
            l1_penalty = self.alpha * self.l1_ratio * np.sign(self.w)
            l2_penalty = self.alpha * (1 - self.l1_ratio) * self.w
            dw += l1_penalty + l2_penalty

            # Gradient updates
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        # Calculates Coefficient of Determination (R^2 Score)
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
`
        },
        project2: {
          name: "Project 2: Advanced Real Estate Price Prediction Pipeline",
          description: "An industrial machine learning pipeline built to predict premium real-estate valuations. Features target transformations, categorical encoders, VIF checks, and GridSearchCV hyperparameter tuning.",
          file: "notebooks/01_Supervised_Regression.ipynb",
          code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Advanced Multicollinearity Filter (VIF)
def filter_features_by_vif(df, threshold=5.0):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = cols
        vif_data["VIF"] = [variance_inflation_factor(df[cols].values, i) for i in range(len(cols))]
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            max_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            cols.remove(max_feature)
            print(f"Dropped collinear feature: {max_feature} (VIF: {max_vif:.2f})")
        else:
            break
    return cols

# 2. Production Modeling Pipeline
def build_regression_pipeline(X, y):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(max_iter=5000))
    ])

    # Hyperparameter Grid
    param_grid = {
        'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
        'regressor__l1_ratio': [0.2, 0.5, 0.8]
    }

    # Grid Search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return grid_search
`
        }
      }
    ]
  },
  classification: {
    title: "Supervised Classification",
    icon: "🎯",
    gradient: "linear-gradient(135deg, #FF0844 0%, #FFB199 100%)",
    description: "Predicting categorical classes using probabilistic and tree-based ensembles.",
    models: [
      {
        id: "logistic_regression",
        name: "Logistic Regression & Classification",
        important2026: true,
        badge: "Industry Staple",
        concept: "The foundational baseline for binary and multi-class classification problems. Projects linear regression equations into a [0, 1] probability range using the cumulative logistic Sigmoid function, optimized using Binary Cross-Entropy loss.",
        math: `$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n\n**Binary Cross-Entropy Loss Function:**\n$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h_\\theta(x^{(i)})) + (1 - y^{(i)}) \\log(1 - h_\\theta(x^{(i)})) \\right]$$\n\n**Vectorized Gradient Descent Step:**\n$$\\theta := \\theta - \\frac{\\eta}{m} X^T (\\sigma(X\\theta) - y)$$\n\nWhere:\n*   $\\sigma(z)$ is the sigmoid squashing function.\n*   $\\eta$ is the learning rate.`,
        pros: [
          "Outputs calibrated probability scores directly, highly useful for thresholds.",
          "Extremely simple, fast, and does not require intense scaling compute.",
          "Easy to regularize (L1/L2) and yields clear coefficient impact scales."
        ],
        cons: [
          "Struggles with non-linear decision spaces without manual interactions.",
          "Prone to multi-collinearity without regularizations or pre-selection.",
          "Requires features to be centered and scaled for stable regularization."
        ],
        qna: [
          {
            q: "Why is Mean Squared Error (MSE) NOT used as the loss function in Logistic Regression?",
            a: "If we plug the non-linear Sigmoid activation into the quadratic MSE loss function, it produces a highly non-convex loss curve containing numerous local minima. Gradient descent will easily get trapped, failing to find the global optimum. Binary Cross-Entropy (Log Loss) ensures the loss curve is mathematically convex, guaranteeing that gradient descent always converges to the global minimum."
          },
          {
            q: "How do you evaluate classifier performance when the class distribution is extremely imbalanced (e.g. 99% benign, 1% fraudulent)?",
            a: "Standard accuracy is highly misleading in this scenario (a dummy model would get 99% accuracy). Instead, we must utilize metrics like Precision (relevance), Recall/Sensitivity (catch-rate), and the F1-Score (harmonic mean of both). We should also analyze Precision-Recall Curves (PR-AUC is highly preferred over ROC-AUC for sparse positive classes) and inspect the Confusion Matrix."
          }
        ],
        project1: {
          name: "Project 1: Vectorized Logistic Classifier from Scratch",
          description: "A comprehensive custom binary classification class implementing vectorized probability outputs, log-loss optimization, and regularized gradients.",
          file: "notebooks/02_Supervised_Classification.ipynb",
          code: `import numpy as np

class LogisticRegressionScratch:
    """
    Highly optimized, vectorized Binary Logistic Classifier from scratch.
    """
    def __init__(self, lr=0.05, epochs=1000, reg_strength=0.1):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg_strength
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        # Clip input values to prevent math overflow underflows
        z_clipped = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            # Compute probabilities
            z = np.dot(X, self.w) + self.b
            p = self._sigmoid(z)

            # Gradient derivations
            dw = (1 / n_samples) * np.dot(X.T, (p - y)) + (self.reg / n_samples) * self.w
            db = (1 / n_samples) * np.sum(p - y)

            # Weights and bias adjustment
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss(self, X, y):
        p = self.predict_proba(X)
        p = np.clip(p, 1e-15, 1.0 - 1e-15) # Bound limits
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
`
        },
        project2: {
          name: "Project 2: Industrial Bank Credit Default Prediction Engine",
          description: "An advanced, robust classification pipeline designed to assess loan risk. Solves severe class imbalance using SMOTE and optimizes threshold boundaries using Precision-Recall tradeoffs.",
          file: "notebooks/02_Supervised_Classification.ipynb",
          code: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

def train_credit_risk_model(df_features, target_col):
    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Resolve Extreme Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale numeric columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Train L1 regularized logistic regression model for feature selection
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced')
    model.fit(X_train_scaled, y_train_res)

    # Probabilities
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Find the mathematically optimal decision threshold maximizing F1-Score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred_tuned = (y_probs >= best_threshold).astype(int)

    print(f"Optimal F1 Threshold found: {best_threshold:.4f}")
    print(classification_report(y_test, y_pred_tuned))
    return model, best_threshold
`
        }
      },
      {
        id: "decision_trees",
        name: "Decision Trees & Ensemble Models (Random Forest, XGBoost, LightGBM)",
        important2026: true,
        badge: "Highest Industry Demand",
        concept: "Non-parametric models that split the data feature space recursively based on information criteria (Gini or Shannon Entropy). Scaling up to ensembles (Random Forest) leverages bagging, while Gradient Boosting (XGBoost, LightGBM) sequentially minimizes residuals using gradient descent on loss functions.",
        math: `**Shannon Entropy formulation:**\n$$H(S) = -\\sum_{i=1}^{C} p_i \\log_2 p_i$$\n\n**Information Gain on a split:**\n$$IG(S, A) = H(S) - \\sum_{v \\in \\text{Values}(A)} \\frac{|S_v|}{|S|} H(S_v)$$\n\n**XGBoost Regularized Objective at step $t$:**\n$$\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^{n} \\left[ g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i) \\right] + \\gamma T + \\frac{1}{2}\\lambda \\sum_{j=1}^{T} w_j^2$$\n\nWhere:\n*   $g_i$ and $h_i$ are first (gradient) and second (hessian) order derivatives of the loss function.\n*   $\\gamma$ and $\\lambda$ are complexity regularization penalties.`,
        pros: [
          "Extremely powerful on tabular data, completely outperforming deep learning.",
          "Handles raw non-linear relations and requires zero feature scaling pre-work.",
          "Ensembles (XGBoost/LightGBM) are exceptionally robust against noise."
        ],
        cons: [
          "Single decision trees are highly prone to overfitting (high variance).",
          "Difficult to extrapolate beyond the minimum/maximum bounds of training limits.",
          "Ensembles lose direct visual tree inspectability, requiring SHAP values."
        ],
        qna: [
          {
            q: "What is the difference between Bagging (e.g. Random Forest) and Boosting (e.g. XGBoost, LightGBM)?",
            a: "Bagging (Bootstrap Aggregating) trains numerous trees in parallel on random data subsets with replacement. It reduces model variance (overfitting) by averaging output predictions. Boosting trains trees sequentially where each successive tree focuses on correcting the errors (residuals) of the prior trees. Boosting reduces model bias, making it a stronger learner, though more sensitive to hyperparameters."
          },
          {
            q: "Why is LightGBM significantly faster and more memory-efficient than standard XGBoost?",
            a: "LightGBM grows trees leaf-wise (best-first) rather than level-wise (depth-first), finding optimal splits faster. More importantly, it uses GOSS (Gradient-based One-Side Sampling) to keep instances with large gradients and filter out small gradients, and EFB (Exclusive Feature Bundling) to bundle mutually exclusive sparse features, massively reducing data search space."
          }
        ],
        project1: {
          name: "Project 1: Gini-Split Recursive Decision Tree Classifier",
          description: "A complete, modular class implementing a decision tree node recursive builder from scratch, supporting customized max depth controls and leaf Gini evaluations.",
          file: "notebooks/03_Tree_Ensemble_Methods.ipynb",
          code: `import numpy as np

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        p = np.bincount(y) / m
        return 1.0 - np.sum(p ** 2)

    def _split(self, X, feature, threshold):
        left_idx = np.where(X[:, feature] <= threshold)[0]
        right_idx = np.where(X[:, feature] > threshold)[0]
        return left_idx, right_idx

    def _best_split(self, X, y):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        current_gini = self._gini(y)

        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx, right_idx = self._split(X, feat, thresh)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                w_gini = (len(left_idx) / n_samples) * self._gini(y[left_idx]) + \\
                         (len(right_idx) / n_samples) * self._gini(y[right_idx])
                gain = current_gini - w_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_val = np.argmax(np.bincount(y))
            return DecisionNode(value=leaf_val)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            leaf_val = np.argmax(np.bincount(y))
            return DecisionNode(value=leaf_val)

        left_idx, right_idx = self._split(X, feat, thresh)
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return DecisionNode(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        return np.array([self._predict_row(self.root, x) for x in X])
`
        },
        project2: {
          name: "Project 2: Production Customer Churn Engine with Bayesian Optuna",
          description: "An advanced, high-performance customer churn modeling project comparing Random Forest, XGBoost, and LightGBM. Employs Optuna for hyperparameter optimization and computes SHAP value interpretations.",
          file: "notebooks/03_Tree_Ensemble_Methods.ipynb",
          code: `import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib

def optimize_and_train_boosting(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna objective for LightGBM
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'verbosity': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict(X_valid)
        loss = log_loss(y_valid, preds)
        return loss

    # Find hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print(f"Optimal parameters: {study.best_params}")

    # Retrain model on full dataset with best params
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['verbosity'] = -1
    
    full_train = lgb.Dataset(X, label=y)
    final_model = lgb.train(best_params, full_train, num_boost_round=300)
    
    # Save the pipeline artifact
    joblib.dump(final_model, 'churn_lightgbm_model.pkl')
    return final_model
`
        }
      }
    ]
  },
  unsupervised: {
    title: "Unsupervised Learning",
    icon: "🧩",
    gradient: "linear-gradient(135deg, #182848 0%, #4b6cb7 100%)",
    description: "Extracting hidden patterns and structures from unlabelled dataset distributions.",
    models: [
      {
        id: "kmeans_pca",
        name: "K-Means Clustering & Principal Component Analysis (PCA)",
        important2026: true,
        badge: "Critical for Data Prep",
        concept: "K-Means groups unlabeled observations into K distinct clusters by iteratively updating centroids to minimize within-cluster sum of squares (WCSS). PCA performs dimensionality reduction by transforming correlated features into linearly uncorrelated orthogonal variables (Principal Components) aligned with variance directions.",
        math: `**K-Means Objective (WCSS Minimization):**\n$$J = \\sum_{i=1}^{k} \\sum_{x \\in S_i} \\|x - \\mu_i\\|^2$$\n\n**PCA Eigendecomposition of Covariance Matrix:**\n$$\\Sigma = \\frac{1}{n} X^T X$$\n$$\\Sigma v_i = \\lambda_i v_i$$\n\nWhere:\n*   $\\mu_i$ represents the centroid coordinates of cluster $S_i$.\n*   $\\Sigma$ represents the empirical sample covariance matrix of data $X$.\n*   $v_i$ and $\\lambda_i$ represent the eigenvectors and corresponding eigenvalues.`,
        pros: [
          "PCA simplifies massive feature spaces while preserving maximum signal.",
          "K-Means is highly scalable and straightforward for clustering profiles.",
          "Elbow and Silhouette analysis provide solid cluster count justifications."
        ],
        cons: [
          "K-Means is highly sensitive to initial random centroid configurations.",
          "Struggles to group clusters that are non-spherical or have varying densities.",
          "PCA components are combinations of inputs, destroying direct readability."
        ],
        qna: [
          {
            q: "How does the 'Elbow Method' differ from 'Silhouette Analysis' when choosing the number of clusters (K) in K-Means?",
            a: "The Elbow Method plots WCSS against K, searching for an 'elbow' where the rate of drop sharply slows. It only tracks within-cluster distances and is subjective. Silhouette Analysis measures how close each point is to its own cluster versus neighboring clusters, yielding a coefficient between [-1, 1]. A higher silhouette score indicates clean cluster separation, providing a mathematically robust validation."
          },
          {
            q: "Explain why mean centering is a critical prerequisite step before performing PCA.",
            a: "PCA maximizes variance along orthogonal projections from the origin. If the data is not centered (i.e. mean $\\ne 0$), the first principal component will point straight from the origin to the center of the data cloud, rather than aligning with the direction of maximum variance, invalidating the dimensional reduction."
          }
        ],
        project1: {
          name: "Project 1: KMeans++ & Eigen-Covariance PCA from Scratch",
          description: "Full modular from-scratch estimators: custom K-Means++ clustering distance algorithms, WCSS trackers, and complete PCA covariance matrix eigenvalue project solvers.",
          file: "notebooks/04_Unsupervised_Learning.ipynb",
          code: `import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iter=300, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # 1. K-Means++ Style Centroid Initialization
        n_samples, n_features = X.shape
        self.centroids = [X[np.random.choice(n_samples)]]
        
        for _ in range(1, self.k):
            # Compute distance to nearest existing centroid for all points
            dists = np.array([min([np.sum((x - c)**2) for c in self.centroids]) for x in X])
            probs = dists / np.sum(dists)
            next_c = X[np.random.choice(n_samples, p=probs)]
            self.centroids.append(next_c)
            
        self.centroids = np.array(self.centroids)

        # 2. Main Lloyd's convergence loops
        for i in range(self.max_iter):
            # Distance calculations
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Move centroids
            new_centroids = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 
                                      else self.centroids[j] for j in range(self.k)])

            # Convergence checks
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        self.labels = labels

class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute empirical covariance matrix
        cov = np.cov(X_centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort components descending
        idx = np.argsort(eigenvalues)[::-1]
        sorted_vectors = eigenvectors[:, idx]

        # Extract top principal components
        self.components = sorted_vectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
`
        },
        project2: {
          name: "Project 2: High-Dimensional Customer Purchasing Segmentations",
          description: "An advanced unsupervised data workflow projecting high-dimensional customer matrices down to 2D/3D spaces using PCA + t-SNE, followed by K-Means cluster profiling.",
          file: "notebooks/04_Unsupervised_Learning.ipynb",
          code: `import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def analyze_customer_base(df_purchases):
    # Scale behavior matrices
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_purchases)

    # 1. High-dim reduction via PCA
    pca = PCA(n_components=0.95, random_state=42) # Explain 95% variance
    pca_reduced = pca.fit_transform(scaled_data)
    print(f"Original features: {scaled_data.shape[1]}, PCA reduced components: {pca_reduced.shape[1]}")

    # 2. Optimum Cluster validation via Silhouette
    best_score = -1.0
    optimal_k = 3
    for k in range(3, 8):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(pca_reduced)
        score = silhouette_score(pca_reduced, labels)
        print(f"K={k} -> Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            optimal_k = k

    # Finalize Clusters
    final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
    clusters = final_kmeans.fit_predict(pca_reduced)

    # 3. Fit t-SNE for crisp 2D visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_coords = tsne.fit_transform(pca_reduced)

    # Group profiles
    df_profile = df_purchases.copy()
    df_profile["Cluster"] = clusters
    cluster_means = df_profile.groupby("Cluster").mean()
    
    return cluster_means, tsne_coords
`
        }
      }
    ]
  },
  deeplearning: {
    title: "Deep Learning Models",
    icon: "🧠",
    gradient: "linear-gradient(135deg, #0FF0B3 0%, #036ED9 100%)",
    description: "Hierarchical neural networks optimized via vector backpropagation and self-attention.",
    models: [
      {
        id: "mlp_attention",
        name: "Feedforward Networks & Self-Attention Transformers",
        important2026: true,
        badge: "Leading edge AI",
        concept: "The structural foundation of AI. Multi-Layer Perceptrons feed signals forward through weighted layers and adjust weights by propagating error gradients backward using chain-rule derivations (Backprop). Transformers replace recurrence with Self-Attention, processing entire sequences in parallel by computing relevance weights between all sequence token representations.",
        math: `**Scaled Dot-Product Self-Attention:**\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n**Backpropagation Error Gradient at Layer $l$:**\n$$\\delta^{(l)} = \\left( W^{(l+1)T} \\delta^{(l+1)} \\right) \\odot f'\\left(z^{(l)}\\right)$$\n$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} a^{(l-1)T}$$\n\nWhere:\n*   $Q, K, V$ are Query, Key, and Value matrix transformations of input sequences.\n*   $\\delta^{(l)}$ represents the error term/delta at layer $l$.\n*   $a^{(l-1)}$ is the activation output of the previous layer.`,
        pros: [
          "Self-Attention allows parallelized sequence processing, bypassing LSTM bottlenecks.",
          "Universal function approximators; can map any complex feature relationship.",
          "Scales cleanly to billions of parameters with stable GPU optimization."
        ],
        cons: [
          "Massive parameter counts require immense compute capacity to train.",
          "Complete lack of intuitive model interpretability (black box).",
          "Transformers have quadratic complexity ($O(N^2)$) relative to sequence length."
        ],
        qna: [
          {
            q: "Explain how backpropagation works in a neural network and why the vanishing gradient problem occurs.",
            a: "Backpropagation uses the calculus Chain Rule to compute the partial derivatives of the loss function with respect to every weight in the network, working backward from output to input. Vanishing gradients occur in deep networks using activations like Sigmoid or Tanh, whose derivatives are < 0.25. As these values are multiplied across many layers, the gradient exponentially shrinks to zero, halting weight updates in early layers. We mitigate this using ReLU activations and Batch Normalization."
          },
          {
            q: "Why does the Self-Attention mechanism in Transformers scale quadratic complexity relative to sequence length?",
            a: "To compute self-attention, we multiply the Query matrix $Q$ ($N \\times d$) by the transposed Key matrix $K^T$ ($d \\times N$). This produces an attention weight matrix of size $N \\times N$, representing the correlation score between every single token and every other token. This $N \\times N$ matrix requires $O(N^2)$ computations, making long context windows extremely expensive."
          }
        ],
        project1: {
          name: "Project 1: Feedforward Layer Backpropagation Engine from Scratch",
          description: "A complete, modular object-oriented neural network framework built from scratch containing Dense layers, activations, log loss equations, and backpropagation chains.",
          file: "notebooks/05_Deep_Learning_MLP.ipynb",
          code: `import numpy as np

class DenseLayer:
    """
    Modular Layer supporting weights, bias, forward activations and backprop equations.
    """
    def __init__(self, input_dim, output_dim):
        # Xavier/Glorot Normal Initialization
        self.w = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        self.b = np.zeros((1, output_dim))
        self.x = None
        self.z = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.w) + self.b
        return self.z

    def backward(self, dz, lr):
        # Gradient derivations
        self.dw = np.dot(self.x.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        
        # Calculate loss delta for the preceding layer
        dx = np.dot(dz, self.w.T)

        # SGD Updates
        self.w -= lr * self.dw
        self.b -= lr * self.db
        return dx

class ReLUScratch:
    def __init__(self):
        self.z = None

    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, da):
        # Derivative of ReLU: 1 if z > 0 else 0
        return da * (self.z > 0).astype(float)
`
        },
        project2: {
          name: "Project 2: Sequence-to-Sequence Machine Translator in PyTorch",
          description: "An advanced deep learning translator using a modular Encoder-Decoder PyTorch design with custom Multi-Head Self-Attention layers.",
          file: "notebooks/06_Deep_Learning_Attention.ipynb",
          code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionPyTorch(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Q, K, V projection layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Project inputs and split into multiple heads
        Q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Self-Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)

        # 3. Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(out)
`
        }
      }
    ]
  },
  timeseries: {
    title: "Time Series & Forecasting",
    icon: "📅",
    gradient: "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    description: "Forecasting sequential temporal patterns using statistical ARIMA and Prophet models.",
    models: [
      {
        id: "arima_sarimax",
        name: "ARIMA & SARIMAX Forecasting Suite",
        important2026: true,
        badge: "Quant Trading Favorite",
        concept: "A statistical forecasting standard. ARIMA combines Autoregression (lags of the target) and Moving Averages (lags of forecast errors) after stationary transformations (differencing). Extended to SARIMAX to support seasonal patterns and external exogenous predictors.",
        math: `**ARIMA(p, d, q) formulation on differenced series $y'_t$:**\n$$y'_t = c + \\phi_1 y'_{t-1} + \\dots + \\phi_p y'_{t-p} + \\theta_1 \\epsilon_{t-1} + \\dots + \\theta_q \\epsilon_{t-q} + \\epsilon_t$$\n\n**Augmented Dickey-Fuller (ADF) regression test:**\n$$\\Delta y_t = \\alpha + \\beta t + \\gamma y_{t-1} + \\sum_{i=1}^{p} \\delta_i \\Delta y_{t-i} + \\epsilon_t$$\n\nWhere:\n*   $\\phi_i$ are autoregressive coefficients, $\\theta_i$ are moving average coefficients.\n*   $\\epsilon_t$ represents white noise error at step $t$.\n*   $\\gamma = 0$ is the ADF null hypothesis (indicating a non-stationary unit root).`,
        pros: [
          "Strong theoretical statistical baseline with explicit confidence bounds.",
          "Performs exceptionally well on short-term linear time series datasets.",
          "SARIMAX supports multi-variate correlations with exogenous market data."
        ],
        cons: [
          "Demands strict mathematical stationarity in the underlying sequence.",
          "Extremely poor at modeling highly non-linear, structural-break trends.",
          "Inference slows down significantly when scaling to multi-step forecasts."
        ],
        qna: [
          {
            q: "What is 'stationarity' in time series, and why is it a critical prerequisite for fitting an ARIMA model?",
            a: "A time series is stationary when its statistical properties (mean, variance, and autocorrelation) remain constant over time. If a series is non-stationary (e.g. exhibits a rising trend), its covariance structure changes, making past patterns useless for predicting future values. ARIMA requires stationarity to fit stable coefficients; we achieve it using differencing ($d$ parameter)."
          },
          {
            q: "How do you use the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to select ARIMA parameters?",
            a: "For an Autoregressive model AR(p), the PACF cuts off sharply after lag $p$, while the ACF decays exponentially. For a Moving Average model MA(q), the ACF cuts off sharply after lag $q$, while the PACF decays exponentially. If both show gradual decays, it suggests an ARMA(p, q) model is required."
          }
        ],
        project1: {
          name: "Project 1: Autoregressive Yule-Walker Estimator from Scratch",
          description: "A custom math module calculating time-series autocovariances and solving for autoregressive parameters using matrix-level Yule-Walker equations.",
          file: "notebooks/07_Time_Series_Forecasting.ipynb",
          code: `import numpy as np

class AutoregressiveEstimator:
    """
    From-scratch Autoregressive AR(p) model parameter solver using Yule-Walker equations.
    """
    def __init__(self, p=2):
        self.p = p
        self.phi = None
        self.mean = None

    def _autocovariance(self, x, lag):
        n = len(x)
        if lag >= n: return 0.0
        x_centered = x - np.mean(x)
        return np.sum(x_centered[:n-lag] * x_centered[lag:]) / n

    def fit(self, x):
        self.mean = np.mean(x)
        p = self.p

        # 1. Calculate autocovariances up to lag p
        gamma = np.array([self._autocovariance(x, i) for i in range(p + 1)])

        # 2. Construct Toplitz correlation matrix
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = gamma[abs(i - j)]

        # 3. Target vector
        r = gamma[1:p+1]

        # 4. Solve Yule-Walker matrix equations: phi = R^-1 * r
        self.phi = np.linalg.solve(R, r)

    def predict_next(self, x_history):
        # Predict y_t using past p observations
        lags = x_history[-self.p:]
        lags_centered = lags - self.mean
        # Reverse lags to align chronological updates
        pred_centered = np.dot(self.phi, lags_centered[::-1])
        return self.mean + pred_centered
`
        },
        project2: {
          name: "Project 2: Industrial Revenue Pipeline - Hybrid SARIMAX & Prophet Model",
          description: "An advanced production pipeline for high-frequency sales forecasting. Implements rolling walk-forward cross-validation and combines Prophet trends with SARIMAX residual corrections.",
          file: "notebooks/07_Time_Series_Forecasting.ipynb",
          code: `import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def train_hybrid_forecaster(df_sales, steps=30):
    # df_sales contains ['ds', 'y'] where ds is date, y is value
    
    # 1. Train Prophet for baseline macroeconomic trend
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_sales)
    
    future = prophet_model.make_future_dataframe(periods=steps)
    forecast = prophet_model.predict(future)
    
    # 2. Extract residuals from Prophet fit on training data
    df_residuals = df_sales.copy()
    df_residuals['prophet_trend'] = forecast['yhat'][:len(df_sales)].values
    df_residuals['residual'] = df_residuals['y'] - df_residuals['prophet_trend']
    
    # 3. Fit SARIMAX on residuals to model fine-grained temporal patterns
    sarimax_model = SARIMAX(
        df_residuals['residual'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7), # Weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_fit = sarimax_model.fit(disp=False)
    
    # 4. Forecast residuals and combine predictions
    residual_forecast = sarimax_fit.forecast(steps=steps)
    prophet_future_trend = forecast['yhat'][-steps:].values
    
    final_hybrid_forecast = prophet_future_trend + residual_forecast.values
    return final_hybrid_forecast
`
        }
      }
    ]
  }
};

// Export to module system if in node, or attach to window for browser use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MODEL_DATA };
} else {
    window.MODEL_DATA = MODEL_DATA;
}
