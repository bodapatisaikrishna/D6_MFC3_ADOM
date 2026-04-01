
# ADOM: ADMM-Based Optimization Model for Stripe Noise Removal

## Overview
This repository contains a MATLAB implementation of the **ADMM-Based Optimization Model (ADOM)** for removing stripe noise from images. The project provides robust algorithms to tackle various types of structured noise, including vertical, horizontal, diagonal, and complex multi-directional stripes. It leverages the Alternating Direction Method of Multipliers (ADMM) and Fast Fourier Transform (FFT) for efficient, high-quality image restoration.

## Key Features
* **Multi-Directional Support:** Dedicated functions for vertical, horizontal, and diagonal stripe removal.
* **Unified Pipeline:** A comprehensive function (`All-destripe.mlx`) capable of sequentially processing and removing stripes in any specified direction.
* **Synthetic Noise Generation:** Built-in scripts to simulate stripe noise on clean images for testing and benchmarking.
* **Efficient Solvers:** Utilizes FFT in the frequency domain to efficiently solve ADMM subproblems.
* **Adaptive Weight Control:** Implements evidence-based starting point control and momentum-based step-size adjustments for faster convergence.

## File Structure
* **`ADOM_vert.mlx`**: Core function to remove vertical stripe noise.
* **`ADOM_hori.mlx`**: Core function to remove horizontal stripe noise.
* **`ADOM_diag.mlx`**: Core function to remove diagonal stripe noise.
* **`ADOM_2D.mlx`**: Pipeline for bidirectional (vertical + horizontal) stripe removal.
* **`All-destripe.mlx`**: The unified, multi-directional ADOM function. Accepts a `direction` argument (`'vertical'`, `'horizontal'`, or `'diagonal'`).
* **`ADOM_stripe.mlx`**: Script to add synthetic vertical stripe noise to a clean image for evaluation.
* **`ADOM_striperemoval.mlx`**: End-to-end script demonstrating the loading of a striped image, processing it through ADOM, and saving the destriped output.

## Prerequisites
* **MATLAB** (R2020a or newer recommended)
* Image Processing Toolbox

## Usage

### 1. Basic Destriping (Unified Function)
To remove noise from an image using the unified function found in `All-destripe.mlx`:

```matlab
% Read the noisy image
O_striped = im2double(rgb2gray(imread('noisy_image.jpg')));

% Set Optimization Parameters
lambda1 = 0.05; % Regularization parameter 1
lambda2 = 0.1;  % Regularization parameter 2
rho1 = 1;       % Penalty parameter 1
rho2 = 1;       % Penalty parameter 2
rho3 = 1;       % Penalty parameter 3
p = 10;         % Acceleration threshold
tol = 1e-4;     % Convergence tolerance
max_iter = 200; % Maximum iterations

% Run ADOM (specify 'vertical', 'horizontal', or 'diagonal')
destriped_img = ADOM(O_striped, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'vertical');

% Display the result
imshow(destriped_img);
title('Destriped Image');
