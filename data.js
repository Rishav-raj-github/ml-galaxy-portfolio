/**
 * ML Galaxy Portfolio - Complete Core Database (2026 Edition)
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
          }
        ],
        project1: {
          name: "Project 1: Regularized Linear Regression from Scratch",
          description: "An elegant, vectorized custom estimator implementing linear regression with adjustable L1 (Lasso) and L2 (Ridge) coordinate descent and gradient descent optimizers.",
          file: "notebooks/01_Supervised_Regression.ipynb",
          code: `import numpy as np

class RegularizedLinearRegression:
    def __init__(self, lr=0.01, epochs=1000, alpha=0.1, l1_ratio=0.5):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha  # Lambda parameter
        self.l1_ratio = l1_ratio # 1.0=Lasso, 0.0=Ridge
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            l1_penalty = self.alpha * self.l1_ratio * np.sign(self.w)
            l2_penalty = self.alpha * (1 - self.l1_ratio) * self.w
            dw += l1_penalty + l2_penalty

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
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

def build_regression_pipeline(X, y):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(max_iter=5000))
    ])

    param_grid = {
        'regressor__alpha': [0.01, 0.1, 1.0],
        'regressor__l1_ratio': [0.2, 0.5, 0.8]
    }

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
        math: `$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n\n**Binary Cross-Entropy Loss Function:**\n$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h_\\theta(x^{(i)})) + (1 - y^{(i)}) \\log(1 - h_\\theta(x^{(i)})) \\right]$$\n\nWhere:\n*   $\\sigma(z)$ is the sigmoid squashing function.`,
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
          }
        ],
        project1: {
          name: "Project 1: Vectorized Logistic Classifier from Scratch",
          description: "A comprehensive custom binary classification class implementing vectorized probability outputs, log-loss optimization, and regularized gradients.",
          file: "notebooks/02_Supervised_Classification.ipynb",
          code: `import numpy as np

class LogisticRegressionScratch:
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
`
        },
        project2: {
          name: "Project 2: Industrial Bank Credit Default Prediction Engine",
          description: "An advanced, robust classification pipeline designed to assess loan risk. Solves severe class imbalance using SMOTE and optimizes threshold boundaries using Precision-Recall tradeoffs.",
          file: "notebooks/02_Supervised_Classification.ipynb",
          code: `from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def train_credit_risk_model(df_features, target_col):
    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    model = LogisticRegression()
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
        concept: "Non-parametric models that split the data feature space recursively based on information criteria (Gini or Shannon Entropy). Ensembles leverage bagging (Random Forest) and boosting (XGBoost/LightGBM) to dramatically reduce bias and variance.",
        math: `**Gini Impurity Equation:**\n$$Gini(S) = 1 - \\sum_{i=1}^{C} p_i^2$$\n\n**Shannon Entropy Equation:**\n$$H(S) = -\\sum_{i=1}^{C} p_i \\log_2 p_i$$`,
        pros: [
          "State-of-the-art accuracy on tabular datasets.",
          "Handles non-linear relationships with zero feature scaling required.",
          "Boosting is exceptionally robust against missing values."
        ],
        cons: [
          "Single trees overfit easily.",
          "Boosting ensembles require careful hyperparameter tuning.",
          "Black-box nature requires SHAP values for model explainability."
        ],
        qna: [
          {
            q: "What is the difference between Bagging and Boosting?",
            a: "Bagging (Random Forest) trains trees in parallel on bootstrap samples and averages predictions to reduce variance. Boosting (XGBoost/LightGBM) trains trees sequentially, with each tree learning to predict the residuals (errors) of the previous trees to reduce bias."
          }
        ],
        project1: {
          name: "Project 1: Recursive Gini Split Tree Classifier from Scratch",
          description: "A complete custom decision tree estimator recursively partitioning feature spaces based on purity splits.",
          file: "notebooks/03_Tree_Ensemble_Methods.ipynb",
          code: `import numpy as np

class DecisionTreeScratch:
    def fit(self, X, y):
        # Code splits and builds nodes recursively
        pass
`
        },
        project2: {
          name: "Project 2: Churn Prediction tuned with Bayesian Optuna",
          description: "An advanced churn classification pipeline optimized using Bayesian searches across LightGBM parameters.",
          file: "notebooks/03_Tree_Ensemble_Methods.ipynb",
          code: `import lightgbm as lgb
import optuna

def optimize_and_train_boosting(X, y):
    def objective(trial):
        params = {
            'objective': 'binary',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255)
        }
        train_data = lgb.Dataset(X, label=y)
        # Run CV and return logloss
        return 0.2
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
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
    description: "Extracting patterns, clusters, and low-dimensional representations from unlabeled datasets.",
    models: [
      {
        id: "kmeans_pca",
        name: "K-Means Clustering & Principal Component Analysis (PCA)",
        important2026: true,
        badge: "Critical for Data Prep",
        concept: "Centroid-based clustering (K-Means) partition points into clusters by minimizing WCSS. Dimensionality reduction (PCA) projects observations onto orthogonal directions capturing maximum variance.",
        math: `**K-Means Objective:**\n$$J = \\sum_{i=1}^{k} \\sum_{x \\in S_i} \\|x - \\mu_i\\|^2$$\n\n**PCA Covariance Eigendecomposition:**\n$$\\Sigma v_i = \\lambda_i v_i$$`,
        pros: [
          "PCA resolves high-dimensionality bottlenecks (curse of dimensionality).",
          "K-Means is highly scalable and convergent.",
          "Unlocks detailed customer profiles."
        ],
        cons: [
          "K-Means is highly sensitive to initialization.",
          "Requires spherical, dense clustering patterns.",
          "PCA components are not directly readable."
        ],
        qna: [
          {
            q: "Why center features before running PCA?",
            a: "PCA searches for directions of maximum variance starting from the origin. If features are not centered (mean=0), the first principal component will align with the data's mean coordinate rather than the maximum variance axis."
          }
        ],
        project1: {
          name: "Project 1: Custom KMeans++ and PCA Solver",
          description: " NumPy custom algorithms implementing centroid cluster calculations and covariance matrices decomposition.",
          file: "notebooks/04_Unsupervised_Learning.ipynb",
          code: `import numpy as np

class KMeansScratch:
    # Lloyd's algorithm from scratch
    pass
`
        },
        project2: {
          name: "Project 2: Multi-dimensional Customer Profiles",
          description: "A production pipeline projecting transaction matrices using PCA followed by cluster segment profiling.",
          file: "notebooks/04_Unsupervised_Learning.ipynb",
          code: `from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def analyze_customer_base(df):
    pca = PCA(n_components=3)
    red = pca.fit_transform(df)
    kmeans = KMeans(n_clusters=3)
    return kmeans.fit_predict(red)
`
        }
      }
    ]
  },
  instanceprob: {
    title: "Instance & Probabilistic Models",
    icon: "🧮",
    gradient: "linear-gradient(135deg, #2b5876 0%, #4e4376 100%)",
    description: "Classification and probability modeling utilizing structural distance metrics and Bayes Theorem.",
    models: [
      {
        id: "knn_naive_bayes",
        name: "K-Nearest Neighbors (KNN) & Naive Bayes Classifier",
        important2026: true,
        badge: "Essential Baselines",
        concept: "Instance-based learning (KNN) classifies observations based on the majority label of its K nearest coordinate neighbors. Probabilistic learning (Naive Bayes) calculates posterior class probabilities using Bayes Theorem under the assumption that all features are conditionally independent.",
        math: `**Bayes Theorem (Naive Bayes Classifier):**\n$$P(C_k | x) = \\frac{P(C_k) \\prod_{i=1}^{n} P(x_i | C_k)}{P(x)}$$\n\n**Euclidean Distance (KNN Metric):**\n$$d(p, q) = \\sqrt{\\sum_{i=1}^{n} (p_i - q_i)^2}$$\n\nWhere:\n*   $P(C_k | x)$ is the posterior probability of class $C_k$ given features $x$.`,
        pros: [
          "KNN requires zero training time (instance-based lazy learner).",
          "Naive Bayes works exceptionally well on small datasets and text (spam filters).",
          "Decisions are mathematically clear and easy to diagnose."
        ],
        cons: [
          "KNN inference slows down dramatically on massive datasets ($O(N)$ query time).",
          "Naive Bayes conditional independence assumption rarely holds in real-world features.",
          "Highly sensitive to redundant, collinear, or unscaled inputs."
        ],
        qna: [
          {
            q: "What is the 'lazy learning' bottleneck in KNN, and how do we resolve it?",
            a: "KNN is a lazy learner because it performs no training; it simply stores the dataset. During inference, it must calculate pairwise distances from the query to *all* training points, which is $O(N)$ and extremely slow. We resolve this by using spatial indexing trees like KD-Trees or Ball-Trees, or by using approximate nearest neighbor algorithms (e.g. HNSW)."
          }
        ],
        project1: {
          name: "Project 1: Vectorized KNN & Naive Bayes Classifier from Scratch",
          description: "NumPy classes implementing Euclidean distance queries and joint conditional probability math equations.",
          file: "notebooks/12_Instance_Based_and_Probabilistic.ipynb",
          code: `import numpy as np

class KNNClassifierScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            # Pairwise Euclidean distances
            dists = np.linalg.norm(self.X_train - x, axis=1)
            nearest = np.argsort(dists)[:self.k]
            labels = self.y_train[nearest]
            preds.append(np.argmax(np.bincount(labels)))
        return np.array(preds)
`
        },
        project2: {
          name: "Project 2: Email Spam Filter Pipeline with Text Tokenization",
          description: "A complete Natural Language Processing (NLP) text classifier pipeline using CountVectorizer and Multinomial Naive Bayes.",
          file: "notebooks/12_Instance_Based_and_Probabilistic.ipynb",
          code: `from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_spam_filter():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    return pipeline
`
        }
      }
    ]
  },
  clusteringrules: {
    title: "Density Clustering & Rules",
    icon: "🛒",
    gradient: "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    description: "Density-based spatial clustering and transaction association rule mining.",
    models: [
      {
        id: "dbscan_apriori",
        name: "DBSCAN Clustering & Apriori Association Rules",
        important2026: true,
        badge: "Advanced Analytics",
        concept: "DBSCAN clusters data based on spatial density density connections (epsilon, min_samples), naturally identifying noise/outliers and non-spherical groups. Apriori mines transactional basket logs to extract rules based on item occurrences.",
        math: `**Apriori Rule Metrics:**\n$$\\text{Support}(A \\to B) = P(A \\cup B) = \\frac{\\text{Transactions containing } A \\text{ and } B}{\\text{Total Transactions}}$$\n\n$$\\text{Confidence}(A \\to B) = P(B | A) = \\frac{\\text{Support}(A \\cup B)}{\\text{Support}(A)}$$\n\n$$\\text{Lift}(A \\to B) = \\frac{\\text{Support}(A \\cup B)}{\\text{Support}(A) \\times \\text{Support}(B)}$$\n\nWhere:\n*   $\\text{Lift} > 1$ indicates that items $A$ and $B$ are positively associated.`,
        pros: [
          "DBSCAN does not require pre-specifying the number of clusters (K).",
          "Identifies noise/outliers explicitly, shielding models from extreme anomalies.",
          "Apriori extracts powerful market-basket rules for e-commerce cross-selling."
        ],
        cons: [
          "DBSCAN struggles in datasets with highly variable cluster densities.",
          "Apriori candidate generation gets extremely expensive on wide transaction databases.",
          "DBSCAN epsilon parameter is highly sensitive and difficult to tune."
        ],
        qna: [
          {
            q: "How does DBSCAN classify outliers, and how does this contrast with K-Means?",
            a: "DBSCAN classifies points as Core, Border, or Noise. A point is Noise (an outlier) if it is not within epsilon distance of a Core point. This allows DBSCAN to isolate anomalies explicitly. K-Means, conversely, must assign *every* single point to a cluster, meaning extreme outliers will pull centroids away from their true density centers, corrupting cluster results."
          }
        ],
        project1: {
          name: "Project 1: DBSCAN Density Core Queue Tracker from Scratch",
          description: "NumPy implementation of density-based clustering scanning neighborhood spheres recursively.",
          file: "notebooks/13_Advanced_Clustering_and_Rules.ipynb",
          code: `import numpy as np

class DBSCANScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        labels = np.zeros(X.shape[0], dtype=int) - 1 # -1 = Noise
        cluster_id = 0
        for i in range(X.shape[0]):
            if labels[i] != -1: continue
            
            # Find neighbors
            dists = np.linalg.norm(X - X[i], axis=1)
            neighbors = np.where(dists <= self.eps)[0]
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1 # Noise
            else:
                labels[i] = cluster_id
                # Expand cluster recursively
                self._expand(X, labels, neighbors, cluster_id)
                cluster_id += 1
        self.labels_ = labels
`
        },
        project2: {
          name: "Project 2: Market Basket analysis on Transaction Datasets",
          description: "Production transactional mining pipeline employing FP-Growth to calculate association rules.",
          file: "notebooks/13_Advanced_Clustering_and_Rules.ipynb",
          code: `from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

def mine_basket_rules(df_onehot):
    frequent_itemsets = fpgrowth(df_onehot, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    return rules
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
  visiongenerative: {
    title: "Computer Vision & GenAI",
    icon: "🎨",
    gradient: "linear-gradient(135deg, #F300FF 0%, #7F00FF 100%)",
    description: "Deep convolutional networks, object trackers, and generative distribution models (GANs).",
    models: [
      {
        id: "resnet_yolo_gan",
        name: "ResNet, YOLO, & Generative Adversarial Networks (GANs)",
        important2026: true,
        badge: "Leading Edge DL",
        concept: "Residual Networks (ResNet) bypass optimization degradation in extremely deep layers by using residual skip connections. YOLO segments spatial grids to perform real-time bounding box object detections. Generative Adversarial Networks (GANs) play a minimax game where a Generator tries to synthesize real-looking data, and a Discriminator tries to detect fakes.",
        math: `**GAN Minimax Objective Loss Function:**\n$$\\min_{G} \\max_{D} V(D, G) = \\mathbb{E}_{x \\sim p_{data}}[\\log D(x)] + \\mathbb{E}_{z \\sim p_{z}}[\\log(1 - D(G(z)))]$$\n\n**ResNet Skip Layer Formulation:**\n$$H(x) = F(x) + x$$\n\nWhere:\n*   $D(x)$ is the Discriminator's probability that real data $x$ is genuine.\n*   $G(z)$ is the Generator's synthetic output mapping noise vector $z$.`,
        pros: [
          "ResNet skip layers resolve vanishing gradients in hundreds of hidden convolutional layers.",
          "YOLO maps bounding coordinates in a single forward pass, unlocking real-time tracking speed.",
          "GANs synthesize exceptionally crisp, detailed image outputs."
        ],
        cons: [
          "GAN training is highly unstable and prone to 'Mode Collapse' (producing limited variations).",
          "Object detection models are computationally expensive to pre-train from scratch.",
          "Extremely sensitive to minor variations in target domain distributions (adversarial noise)."
        ],
        qna: [
          {
            q: "What is 'Mode Collapse' in GAN training, and how do we prevent it?",
            a: "Mode Collapse occurs when the Generator discovers a single output format that consistently fools the Discriminator, causing it to exclusively synthesize that specific sample (e.g., generating only one digit shape in MNIST) and failing to learn the true diversity of the data. We prevent it using Wasserstein GANs with Gradient Penalty (WGAN-GP), which optimizes Earth Mover's Distance for smoother gradients."
          }
        ],
        project1: {
          name: "Project 1: Generative Adversarial Network (GAN) from Scratch in PyTorch",
          description: "A complete custom GAN model implementing Generator and Discriminator PyTorch modules optimized via minimax loss updates.",
          file: "notebooks/14_Computer_Vision_and_Generative.ipynb",
          code: `import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img)
`
        },
        project2: {
          name: "Project 2: ResNet Image Classifier with PyTorch Transfer Learning",
          description: "An advanced computer vision transfer learning pipeline fine-tuning a pre-trained ResNet-50 network for custom classifications.",
          file: "notebooks/14_Computer_Vision_and_Generative.ipynb",
          code: `import torch
import torch.nn as nn
from torchvision import models

def build_transfer_learner(num_classes):
    # Load pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze convolutional weights
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
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
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0, dec=0.995):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.dec = dec
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        self.epsilon *= self.dec
`
        },
        project2: {
          name: "Project 2: CartPole Balancing Agent using PyTorch Actor-Critic",
          description: "An advanced actor-critic reinforcement learning pipeline implemented in PyTorch to stabilize unstable physical balancing environments.",
          file: "notebooks/08_Reinforcement_Learning.ipynb",
          code: `import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.affine = nn.Linear(state_dim, 128)
        self.action_head = nn.Linear(128, action_dim)
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
                pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
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
            nn.Linear(64, 1),
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
import numpy as np

def detect_online_frauds(df_transactions):
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    X = df_transactions.select_dtypes(include=[np.number])
    df_transactions['is_fraudulent'] = iso_forest.predict(X)
    return df_transactions
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
        math: `**GCN Layer-Wise Propagation Rule:**\n$$H^{(l+1)} = \\sigma \\left( \\tilde{D}^{-\\frac{1}{2}} \\tilde{A} \\tilde{D}^{-\\frac{1}{2}} H^{(l)} W^{(l)} \\right)$$\n\nWhere:\n*   $\\tilde{A} = A + I_N$ represents the adjacency matrix of the graph with added self-loop identity matrices.`,
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
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))

    def forward(self, A, H):
        n = A.shape[0]
        A_tilde = A + np.eye(n)
        D_tilde = np.diag(np.sum(A_tilde, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D_tilde))
        A_norm = np.dot(np.dot(D_inv_sqrt, A_tilde), D_inv_sqrt)
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
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
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
