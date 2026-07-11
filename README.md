<img width="981" height="90" alt="image" src="https://github.com/user-attachments/assets/b6383954-569b-4001-a8d0-cd115bb86c14" /><div align="center">
  
# Optimizers from Scratch: Linear Regression

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=flat-square&logo=matplotlib&logoColor=black)](https://matplotlib.org/)

*A dependency-free exploration of how optimization algorithms learn.*

</div>

---

## Table of Contents
- [About the Project](#-about-the-project)
- [Mathematical Foundation](#-mathematical-foundation)
- [Implemented Optimizers](#-implemented-optimizers)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)

---

## About the Project

This repository contains a purely foundational implementation of a Linear Regression model built entirely from scratch. Rather than relying on high-level APIs like `scikit-learn` or `PyTorch`, this project manually handles forward passes, loss calculation, and backpropagation using only **NumPy**.

> 💡 **The primary goal:** To visually and mathematically compare how different optimization algorithms (SGD, Momentum, and Adam) navigate the loss landscape to find optimal weights.

---

## Mathematical Foundation

The notebook relies on the following core principles, implemented via raw code:

**1. The Hypothesis (Linear Model)**
$$y = wx + b$$

**2. The Cost Function (Mean Squared Error)**
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**3. The Gradients (Derivatives for Backpropagation)** 
 $\displaystyle \begin{aligned}
\frac{\partial L}{\partial w} &= -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i) \\[2ex]
\frac{\partial L}{\partial b} &= -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)
\end{aligned}$

---

## Implemented Optimizers

| Optimizer | Description | Key Mechanism |
| :--- | :--- | :--- |
| **SGD** | Stochastic Gradient Descent | The baseline. Updates parameters in the exact opposite direction of the gradient. |
| **Momentum** | SGD with Momentum | Accumulates an exponentially decaying moving average of past gradients to accelerate learning and dampen oscillations. |
| **Adam** | Adaptive Moment Estimation | Combines Momentum (first moment) and RMSProp (second moment) with bias correction for highly efficient, adaptive learning rates. |

---

## Repository Structure

```text
├── optimizers_comparison.ipynb   # Main Jupyter Notebook
├── README.md                     # Project documentation
