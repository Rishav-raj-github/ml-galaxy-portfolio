/**
 * ML Galaxy Portfolio - Expanded Core Database (2026 Edition)
 * Houses comprehensive metadata, LaTeX math formulations, interview Q&As,
 * and high-quality Python code blocks for ALL models in your taxonomy.
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
        z_clipped = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            p = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (p - y)) + (self.reg / n_samples) * self.w
            db = (1 / n_samples) * np.sum(p - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced')
    model.fit(X_train_scaled, y_train_res)
    return model
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
                if len(left_idx) == 0 or len(right_idx) == 0: continue

                w_gini = (len(left_idx) / n_samples) * self._gini(y[left_idx]) + (len(right_idx) / n_samples) * self._gini(y[right_idx])
                gain = current_gini - w_gini
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh
        return split_idx, split_thresh

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
`
        },
        project2: {
          name: "Project 2: Production Customer Churn Engine with Bayesian Optuna",
          description: "An advanced, high-performance customer churn modeling project comparing Random Forest, XGBoost, and LightGBM. Employs Optuna for hyperparameter optimization and computes SHAP value interpretations.",
          file: "notebooks/03_Tree_Ensemble_Methods.ipynb",
          code: `import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def optimize_and_train_boosting(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'verbosity': -1
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        model = lgb.train(params, train_data, valid_sets=[valid_data])
        preds = model.predict(X_valid)
        return log_loss(y_valid, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    return study.best_params
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
    def __init__(self, k=3, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 
                                      else self.centroids[j] for j in range(self.k)])

            if np.linalg.norm(new_centroids - self.centroids) < 1e-4:
                break
            self.centroids = new_centroids
        self.labels = labels
`
        },
        project2: {
          name: "Project 2: High-Dimensional Customer Purchasing Segmentations",
          description: "An advanced unsupervised data workflow projecting high-dimensional customer matrices down to 2D/3D spaces using PCA + t-SNE, followed by K-Means cluster profiling.",
          file: "notebooks/04_Unsupervised_Learning.ipynb",
          code: `import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def analyze_customer_base(df_purchases):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_purchases)

    pca = PCA(n_components=0.95, random_state=42)
    pca_reduced = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(pca_reduced)
    return clusters
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
    def __init__(self, input_dim, output_dim):
        self.w = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        self.b = np.zeros((1, output_dim))
        self.x = None
        self.z = None

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.w) + self.b
        return self.z

    def backward(self, dz, lr):
        dw = np.dot(self.x.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.w.T)
        self.w -= lr * dw
        self.b -= lr * db
        return dx
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

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
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
        gamma = np.array([self._autocovariance(x, i) for i in range(p + 1)])
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = gamma[abs(i - j)]
        self.phi = np.linalg.solve(R, gamma[1:p+1])
`
        },
        project2: {
          name: "Project 2: Industrial Revenue Pipeline - Hybrid SARIMAX & Prophet Model",
          description: "An advanced production pipeline for high-frequency sales forecasting. Implements rolling walk-forward cross-validation and combines Prophet trends with SARIMAX residual corrections.",
          file: "notebooks/07_Time_Series_Forecasting.ipynb",
          code: `from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

def train_hybrid_forecaster(df_sales, steps=30):
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_sales)
    
    future = prophet_model.make_future_dataframe(periods=steps)
    forecast = prophet_model.predict(future)
    
    df_residuals = df_sales.copy()
    df_residuals['prophet_trend'] = forecast['yhat'][:len(df_sales)].values
    df_residuals['residual'] = df_residuals['y'] - df_residuals['prophet_trend']
    
    sarimax_model = SARIMAX(df_residuals['residual'], order=(1, 1, 1))
    sarimax_fit = sarimax_model.fit(disp=False)
    return sarimax_fit
`
        }
      }
    ]
  },
  reinforcementlearning: {
    title: "Reinforcement Learning",
    icon: "🎮",
    gradient: "linear-gradient(135deg, #f857a6 0%, #ff5858 100%)",
    description: "Agent-based learning environments maximizing cumulative rewards through penalties.",
    models: [
      {
        id: "qlearning_ppo",
        name: "Q-Learning & Proximal Policy Optimization (PPO)",
        important2026: true,
        badge: "RL Standard",
        concept: "Value-based systems (Q-Learning) learn an action-value function representing the expected utility of taking an action in a state. Policy-based systems like PPO (used to align ChatGPT/LLMs) optimize policy parameters directly, employing a clipped surrogate objective to prevent destabilizing updates.",
        math: `**Bellman Optimality Equation (Q-Learning Update):**\n$$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$\n\n**PPO Clipped Surrogate Objective:**\n$$L^{CLIP}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t) \\right]$$\n\nWhere:\n*   $\\alpha$ represents the learning step rate, and $\\gamma$ represents the discount factor.\n*   $r_t(\\theta) = \\frac{\\pi_\\theta(a_t | s_t)}{\\pi_{\\theta_{old}}(a_t | s_t)}$ represents the probability ratio.\n*   $\\hat{A}_t$ represents the calculated advantage value at step $t$.`,
        pros: [
          "PPO guarantees stable updates, resolving old policy collapse bottlenecks.",
          "Q-Learning converges reliably on finite-state discrete action environments.",
          "Excellent for gaming, robotics, and reinforcement alignment of LLMs."
        ],
        cons: [
          "Extreme sample inefficiency; requires millions of simulation interactions.",
          "Highly sensitive to reward shaping; poor rewards yield zero alignment.",
          "Continuous action spaces demand actor-critic models, increasing complexity."
        ],
        qna: [
          {
            q: "What is the policy collapse problem in Reinforcement Learning, and how does PPO solve it?",
            a: "Policy collapse occurs in policy gradient methods when a step size is too large, causing the policy parameters to update to a region of extremely low performance. Once the policy collapses, the agent cannot collect useful data, preventing recovery. PPO solves this by clipping the probability ratio $r_t(\\theta)$ within $[1-\\epsilon, 1+\\epsilon]$, strictly limiting how much the new policy can deviate from the old policy."
          }
        ],
        project1: {
          name: "Project 1: Q-Table Gridworld Environment Solver from Scratch",
          description: "A customized discrete Gridworld agent solver implementing tabular Bellman updates, epsilon-greedy exploration, and decay rate metrics.",
          file: "notebooks/08_Reinforcement_Learning.ipynb",
          code: `import numpy as np

class QLearningAgent:
    """
    From-scratch discrete action Q-Learning agent implementing Bellman optimization.
    """
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0, dec=0.995):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.dec = dec
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        # Bellman update formula
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

        # Decay epsilon
        self.epsilon *= self.dec
`
        },
        project2: {
          name: "Project 2: CartPole Balancing Agent using PyTorch Actor-Critic",
          description: "An advanced actor-critic reinforcement learning pipeline implemented in PyTorch to stabilize unstable physical balancing environments.",
          file: "notebooks/08_Reinforcement_Learning.ipynb",
          code: `import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.affine = nn.Linear(state_dim, 128)
        
        # Policy head (Actor)
        self.action_head = nn.Linear(128, action_dim)
        # Value head (Critic)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.affine(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values
`
        }
      }
    ]
  },
  recommendation: {
    title: "Recommendation Systems",
    icon: "🍿",
    gradient: "linear-gradient(135deg, #130CB7 0%, #52E5E7 100%)",
    description: "Personalized filtering models leveraging collaborative matrix factorization.",
    models: [
      {
        id: "collaborative_svd",
        name: "Matrix Factorization & SVD++ Recommendations",
        important2026: true,
        badge: "RecSys Gold Standard",
        concept: "Collaborative filtering predicts a user's preference by analyzing behavioral history across users. Singular Value Decomposition (SVD) factorizes the sparse user-item interaction matrix into lower-dimensional user and item latent embedding vectors, optimized to minimize rating prediction errors.",
        math: `**Latent Factor Rating Prediction:**\n$$\\hat{r}_{u,i} = \\mu + b_u + b_i + p_u^T q_i$$\n\n**SVD Regularized Loss Objective:**\n$$\\min_{p_*, q_*, b_*} \\sum_{(u,i) \\in R} (r_{u,i} - \\hat{r}_{u,i})^2 + \\lambda \\left( b_u^2 + b_i^2 + \\|p_u\\|^2 + \\|q_i\\|^2 \\right)$$\n\nWhere:\n*   $\\mu$ represents the global average rating coefficient.\n*   $b_u$ and $b_i$ represent user and item bias deviations from the average.\n*   $p_u$ and $q_i$ represent latent feature vectors for user $u$ and item $i$.`,
        pros: [
          "Extremely effective at finding hidden latent preferences in user matrices.",
          "Scalable; matrix embeddings run extremely fast during runtimes.",
          "Adding bias coefficients dramatically improves baseline prediction accuracy."
        ],
        cons: [
          "Suffers severely from the Cold Start Problem (new users/items with zero data).",
          "Struggles to model dynamic, highly contextual shifts (e.g. time of day).",
          "Sparse data (99%+ empty matrices) destabilizes factor updates."
        ],
        qna: [
          {
            q: "How does the 'Cold Start Problem' affect Collaborative Filtering systems, and how do we solve it?",
            a: "The Cold Start problem occurs when a new user or item enters the system with zero behavioral ratings. Collaborative filtering cannot calculate similarities or factorize embeddings for them. We solve it using Hybrid Systems: employing Content-Based filtering (using metadata like age, location, genre) for initial recommendations, or serving popular/trending items until interaction history is established."
          }
        ],
        project1: {
          name: "Project 1: SVD Latent Matrix Factorizer from Scratch",
          description: "A complete custom NumPy Singular Value Decomposition (SVD) matrix rating factorizer class implementing regularized Stochastic Gradient Descent.",
          file: "notebooks/09_Recommendation_Systems.ipynb",
          code: `import numpy as np

class SVDFactorizerScratch:
    """
    From-scratch regularized matrix factorization recommender optimized via SGD.
    """
    def __init__(self, n_factors=10, lr=0.005, reg=0.02, epochs=50):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    def fit(self, R, user_ids, item_ids, ratings):
        n_users, n_items = R.shape
        self.mu = np.mean(ratings)
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        for epoch in range(self.epochs):
            for u, i, r in zip(user_ids, item_ids, ratings):
                # Predict rating
                pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred

                # Update bias terms
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # Update latent factors
                pu_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu_old - self.reg * self.Q[i])
`
        },
        project2: {
          name: "Project 2: Neural Collaborative Recommender with PyTorch",
          description: "An advanced Deep Learning Recommendation pipeline using PyTorch user/item Embedding layers and Multi-Layer Perceptron matching networks.",
          file: "notebooks/09_Recommendation_Systems.ipynb",
          code: `import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, latent_dim=16):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, latent_dim)
        self.item_embed = nn.Embedding(n_items, latent_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_indices, item_indices):
        u_lat = self.user_embed(user_indices)
        i_lat = self.item_embed(item_indices)
        x = torch.cat([u_lat, i_lat], dim=-1)
        return self.mlp(x).squeeze()
`
        }
      }
    ]
  },
  anomalydetection: {
    title: "Anomaly Detection",
    icon: "🚨",
    gradient: "linear-gradient(135deg, #F00000 0%, #BEBC88 100%)",
    description: "Identifying out-of-distribution observations and fraudulent sequences.",
    models: [
      {
        id: "isolation_forest",
        name: "Isolation Forest & One-Class SVM",
        important2026: true,
        badge: "Security Standard",
        concept: "Anomaly detection isolates outliers instead of profiling normal data. Isolation Forest recursively partitions the feature space using random splits; anomalies require far fewer partitions to isolate because they reside in sparse regions. One-Class SVM maps normal points into a high-dimensional space and constructs a boundary maximizing margin distance from the origin.",
        math: `**Isolation Forest Path Length Score:**\n$$s(x, n) = 2^{-\\frac{\\mathbb{E}(h(x))}{c(n)}}$$\n\n**Average path length of an unsuccessful binary tree search:**\n$$c(n) = 2\\ln(n - 1) + 0.5772 - \\frac{2(n - 1)}{n}$$\n\nWhere:\n*   $\\mathbb{E}(h(x))$ is the average path length (number of edges/splits) to isolate point $x$.\n*   $c(n)$ is the average search path length across a sample size $n$.\n*   $s \\to 1$ indicates a highly isolated anomaly; $s \\to 0$ indicates a highly dense normal coordinate.`,
        pros: [
          "Linear computational complexity $O(N)$ allows real-time security stream scans.",
          "Requires no training label distributions; completely unsupervised.",
          "Very robust against high-dimensional collinearity noise."
        ],
        cons: [
          "Isolation Forest random feature selection can create artifact boundaries.",
          "Highly sensitive to hyperparameter contamination ratios (expected noise).",
          "Difficult to interpret or explain the specific reasons for an anomaly alert."
        ],
        qna: [
          {
            q: "Why is Isolation Forest significantly more effective than distance-based metrics (like KNN or DBSCAN) for high-dimensional anomaly detection?",
            a: "Distance-based metrics suffer from the 'Curse of Dimensionality'—in high-dimensional spaces, the distance between any two points converges, making distance metrics useless. Isolation Forest relies on recursive partitioning rather than calculating global pairwise distances, which drastically reduces computational overhead ($O(N)$ vs $O(N^2)$) and maintains robust isolation capability in high-dimensional datasets."
          }
        ],
        project1: {
          name: "Project 1: Isolation Forest Spatial Partitioner from Scratch",
          description: "A custom unsupervised spatial isolation partition tree class written in NumPy, tracking sample path lengths.",
          file: "notebooks/10_Anomaly_Detection.ipynb",
          code: `import numpy as np

class IsolationTreeNode:
    def __init__(self, left=None, right=None, split_feat=None, split_val=None, size=None):
        self.left = left
        self.right = right
        self.split_feat = split_feat
        self.split_val = split_val
        self.size = size

class IsolationTreeScratch:
    def fit(self, X, current_depth, max_depth):
        n_samples, n_features = X.shape
        if current_depth >= max_depth or n_samples <= 1:
            return IsolationTreeNode(size=n_samples)

        # Select random feature and split value
        feat = np.random.randint(n_features)
        feat_min, feat_max = X[:, feat].min(), X[:, feat].max()
        if feat_min == feat_max:
            return IsolationTreeNode(size=n_samples)

        val = np.random.uniform(feat_min, feat_max)
        left_idx = np.where(X[:, feat] < val)[0]
        right_idx = np.where(X[:, feat] >= val)[0]

        left_node = self.fit(X[left_idx], current_depth + 1, max_depth)
        right_node = self.fit(X[right_idx], current_depth + 1, max_depth)
        return IsolationTreeNode(left_node, right_node, feat, val)
`
        },
        project2: {
          name: "Project 2: Real-time Credit Card Fraud Streaming Detector",
          description: "A production-grade unsupervised security pipeline using Scikit-Learn Isolation Forest to flag credit card transactions in real-time.",
          file: "notebooks/10_Anomaly_Detection.ipynb",
          code: `from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def detect_online_frauds(df_transactions):
    # Train Isolation Forest with target contamination ratio of 1%
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    # Extract features (e.g. transaction amount, frequency)
    X = df_transactions.select_dtypes(include=[np.number])
    
    # Fit model and predict (-1 = Anomaly, 1 = Normal)
    df_transactions['anomaly_score'] = iso_forest.decision_function(X)
    df_transactions['is_fraudulent'] = iso_forest.predict(X)
    
    frauds = df_transactions[df_transactions['is_fraudulent'] == -1]
    print(f"Online scan completed. Flagged transactions: {len(frauds)}")
    return frauds
`
        }
      }
    ]
  },
  graphml: {
    title: "Graph Machine Learning",
    icon: "🕸️",
    gradient: "linear-gradient(135deg, #3A1C71 0%, #D76D77 50%, #FFAF7B 100%)",
    description: "Learning node embedding representations and predicting network relationships.",
    models: [
      {
        id: "gnn_gcn",
        name: "Graph Neural Networks (GNN, GCN, GraphSAGE)",
        important2026: true,
        badge: "Advanced AI",
        concept: "Graph Convolutional Networks (GCN) extend standard convolutions to non-Euclidean graphical structures. They learn node representations by recursively aggregating embedding feature vectors from adjacent neighboring nodes (message passing), normalized by node degrees.",
        math: `**GCN Layer-Wise Propagation Rule:**\n$$H^{(l+1)} = \\sigma \\left( \\tilde{D}^{-\\frac{1}{2}} \\tilde{A} \\tilde{D}^{-\\frac{1}{2}} H^{(l)} W^{(l)} \\right)$$\n\nWhere:\n*   $\\tilde{A} = A + I_N$ represents the adjacency matrix of the graph with added self-loop identity matrices.\n*   $\\tilde{D}$ represents the diagonal node degree matrix of $\\tilde{A}$ ($\\tilde{D}_{ii} = \\sum_j \\tilde{A}_{ij}$).\n*   $H^{(l)}$ represents the node feature embeddings at layer $l$.\n*   $W^{(l)}$ represents the trainable weights matrix parameter.`,
        pros: [
          "Flawlessly models complex relational databases (social networks, molecules, links).",
          "Permutation-equivariant: invariant to node ordering modifications.",
          "Message passing enables semi-supervised learning with very few labels."
        ],
        cons: [
          "Prone to oversmoothing: deep layers cause node representations to become identical.",
          "Immense computational complexity; scaling to large graphs requires sub-sampling.",
          "Debugging message aggregation anomalies is highly difficult."
        ],
        qna: [
          {
            q: "What is the 'Oversmoothing' bottleneck problem in deep GCNs, and how do we resolve it?",
            a: "Oversmoothing occurs when we stack too many GCN layers. Because each layer aggregates features from immediate neighbors, running $L$ layers means each node aggregates information from its $L$-hop neighborhood. If $L$ is large, nodes aggregate overlapping global information, causing their embedding vectors to converge and become identical, rendering them useless for downstream tasks. We resolve this using residual links, dropout, or keeping GCN depth shallow (2-4 layers)."
          }
        ],
        project1: {
          name: "Project 1: Graph Convolution Message Passing Layer from Scratch",
          description: "A custom mathematical GCN layer written in NumPy, implementing structural adjacency aggregation and weight transformations.",
          file: "notebooks/11_Graph_Machine_Learning.ipynb",
          code: `import numpy as np

class GCNLayerScratch:
    """
    From-scratch custom GCN Layer implementing adjacency normalizations and message passing.
    """
    def __init__(self, in_features, out_features):
        # Glorot weight initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))

    def forward(self, A, H):
        # 1. Add self-loops to Adjacency matrix: A_tilde = A + I
        n = A.shape[0]
        A_tilde = A + np.eye(n)

        # 2. Compute Degree Matrix: D_tilde
        D_tilde = np.diag(np.sum(A_tilde, axis=1))

        # 3. Compute Symmetric Normalization: D^-0.5 * A * D^-0.5
        D_inv_sqrt = np.linalg.inv(np.sqrt(D_tilde))
        A_norm = np.dot(np.dot(D_inv_sqrt, A_tilde), D_inv_sqrt)

        # 4. Message aggregation and linear projection: H = A_norm * H * W
        H_next = np.dot(A_norm, H)
        return np.maximum(0, np.dot(H_next, self.W)) # ReLU Activation
`
        },
        project2: {
          name: "Project 2: Node Classification Pipeline with PyTorch Geometric",
          description: "An advanced semi-supervised graph node classification pipeline built using PyTorch Geometric (PyG) and Graph Convolution networks.",
          file: "notebooks/11_Graph_Machine_Learning.ipynb",
          code: `import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GCN Layer + Activation
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        
        # Second GCN Layer for output logits
        x = self.conv2(x, edge_index)
        return x
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
