/**
 * ML Galaxy Portfolio - High-Performance C++ Quant Solver (2026 Edition)
 * Implements a low-latency closed-form Ordinary Least Squares (OLS) estimator
 * using standard vector matrix arithmetic and Gaussian elimination.
 * Designed for low-latency quantitative trading environments.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

// Matrix type definition
typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

class LowLatencyOLS {
public:
    // Computes transpose of a matrix: X^T
    Matrix transpose(const Matrix& X) {
        int r = X.size();
        int c = X[0].size();
        Matrix XT(c, Vector(r, 0.0));
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                XT[j][i] = X[i][j];
            }
        }
        return XT;
    }

    // Computes multiplication of two matrices: A * B
    Matrix multiply(const Matrix& A, const Matrix& B) {
        int rA = A.size();
        int cA = A[0].size();
        int cB = B[0].size();
        Matrix C(rA, Vector(cB, 0.0));
        
        for (int i = 0; i < rA; ++i) {
            for (int j = 0; j < cB; ++j) {
                for (int k = 0; k < cA; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // Computes multiplication of matrix and vector: A * v
    Vector multiplyVector(const Matrix& A, const Vector& v) {
        int r = A.size();
        int c = A[0].size();
        Vector res(r, 0.0);
        
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                res[i] += A[i][j] * v[j];
            }
        }
        return res;
    }

    // Inverts a square matrix using Gaussian Elimination
    Matrix invert(const Matrix& M) {
        int n = M.size();
        Matrix A = M; // Copy input
        
        // Append Identity Matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i].push_back(i == j ? 1.0 : 0.0);
            }
        }

        // Gaussian elimination
        for (int i = 0; i < n; ++i) {
            double pivot = A[i][i];
            if (std::abs(pivot) < 1e-10) {
                throw std::runtime_error("Singular matrix inversion exception");
            }

            for (int j = 0; j < 2 * n; ++j) {
                A[i][j] /= pivot;
            }

            for (int r = 0; r < n; ++r) {
                if (r == i) continue;
                double factor = A[r][i];
                for (int j = 0; j < 2 * n; ++j) {
                    A[r][j] -= factor * A[i][j];
                }
            }
        }

        // Extract inverse
        Matrix inv(n, Vector(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inv[i][j] = A[i][j + n];
            }
        }
        return inv;
    }

    // Fits OLS coefficients: theta = (X^T X)^-1 X^T y
    Vector fit(const Matrix& X, const Vector& y) {
        Matrix XT = transpose(X);
        Matrix XTX = multiply(XT, X);
        Matrix XTX_inv = invert(XTX);
        Vector XTy = multiplyVector(XT, y);
        return multiplyVector(XTX_inv, XTy);
    }
};

int main() {
    // Synthetic data: y = 2*x0 + 3.5*x1
    Matrix X = {
        {1.0, 2.0},
        {2.0, 1.5},
        {3.0, 4.0},
        {4.0, 3.5}
    };
    Vector y = {9.0, 9.25, 20.0, 20.25};

    LowLatencyOLS ols;
    try {
        Vector theta = ols.fit(X, y);
        std::cout << "--- OLS Quant Solver Successfully Calibrated ---" << std::endl;
        std::cout << "Beta Coeff 0: " << theta[0] << std::endl;
        std::cout << "Beta Coeff 1: " << theta[1] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Solver Error: " << e.what() << std::endl;
    }
    return 0;
}
