# ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Images

A MATLAB implementation of the **ADOM** algorithm for removing stripe noise from remote sensing images (RSI). This project extends the original paper to support **vertical**, **horizontal**, **diagonal**, and **bidirectional** stripe noise removal.

> Based on: *"ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image"*  
> Namwon Kim, Seong-Soo Han, Chang-Sung Jeong — IEEE Access, Vol. 11, 2023  
> DOI: [10.1109/ACCESS.2023.3319268](https://doi.org/10.1109/ACCESS.2023.3319268)

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
   - [Problem Formulation](#problem-formulation)
   - [Objective Function](#objective-function)
   - [Constrained Objective Function](#constrained-objective-function)
   - [Augmented Lagrangian Function](#augmented-lagrangian-function)
3. [Optimization Process — 4 Steps Per Iteration](#optimization-process--4-steps-per-iteration)
   - [Step 1: Weight Control](#step-1-weight-control)
   - [Step 2: Evidence-Based Starting Point Control](#step-2-evidence-based-starting-point-control)
   - [Step 3: Momentum-Based Step-Size Control](#step-3-momentum-based-step-size-control)
   - [Step 4: ADMM Subproblem Solving](#step-4-admm-based-subproblem-solving)
   - [Convergence Check](#convergence-check)
4. [Code File Explanations](#code-file-explanations)
   - [ADOM_vert.mlx](#1-adom_vertmlx--vertical-stripe-removal)
   - [ADOM_hori.mlx](#2-adom_horimlx--horizontal-stripe-removal)
   - [ADOM_diag.mlx](#3-adom_diagmlx--diagonal-stripe-removal)
   - [ADOM_2D.mlx](#4-adom_2dmlx--bidirectional-stripe-removal)
   - [ADOM_stripe.mlx](#5-adom_stripemlx--synthetic-stripe-noise-generator)
   - [ADOM_striperemoval.mlx](#6-adom_striperemovalmlx--load-and-remove-stripe-noise)
   - [All-destripe.mlx](#7-all-destripemlx--unified-multi-direction-removal)
5. [Repository Structure](#repository-structure)
6. [Requirements](#requirements)
7. [Getting Started](#getting-started)
8. [Parameters Reference](#parameters-reference)
9. [Computational Complexity](#computational-complexity)
10. [Citation](#citation)
11. [License](#license)

---

## Overview

Remote sensing images (RSI) acquired from satellites or airborne platforms frequently suffer from **stripe noise** — systematic linear artifacts caused by:
- Differences in detector gain and offset between sensor elements
- Sensor calibration errors during image acquisition
- Physical gaps between detector elements in pushbroom-type sensors

Stripe noise degrades image quality and negatively affects downstream tasks such as land-cover classification, object detection, and environmental monitoring.

**ADOM** removes this noise by modeling the observed image as a sum of a clean image and a stripe noise component, and then solving an optimization problem to separate them. Instead of directly subtracting noise estimates, ADOM **iteratively refines the stripe estimate S** using a combination of adaptive weighting and accelerated ADMM.

The key innovations over prior methods are:
1. **Weighted norms** that adapt at every iteration so stripe noise is accurately detected even when it resembles real image texture
2. **ADMM-based acceleration** using momentum and evidence-based starting points for faster convergence

---

## Mathematical Foundation

### Problem Formulation

The degradation model assumes stripe noise is **additive**:

```
O = D + S
```

- `O in R^(m x n)` — Observed (noisy) image with m rows and n columns
- `D in R^(m x n)` — Desired clean image (unknown)
- `S in R^(m x n)` — Stripe noise component (what we solve for)

Once S is found, the clean image is recovered as:

```
D = O - S
```

This is more effective than directly subtracting a rough noise estimate because we solve for S through a principled optimization that uses structural knowledge about what stripe noise looks like.

---

### Objective Function

The core objective function (for vertical stripe removal) is:

```
argmin_S { ||grad_y S||_1  +  lambda1 * ||grad_x(O - S)||_{wn,1}  +  lambda2 * ||S||_{wg,2,1} }
```

Each of the three terms encodes a different piece of prior knowledge about stripe noise:

---

#### Term 1: `||grad_y S||_1` — Vertical Smoothness of Stripe Noise

**Math:**
```
grad_y S  =  S(i, j) - S(i-1, j)   for each pixel (i,j)

||grad_y S||_1  =  sum_i sum_j  |S(i,j) - S(i-1,j)|
```

**Meaning:**
Vertical stripe noise is constant along columns — a stripe at column j has the same intensity for every row. Therefore, the vertical gradient `grad_y S` should be zero (or very small). Minimizing `||grad_y S||_1` forces S to be piecewise constant in the vertical direction, which is exactly the structure of stripe noise. This prevents S from capturing vertically varying image content like natural edges.

**In MATLAB code (ADOM_vert.mlx):**
```matlab
grady_S = S - circshift(S, [1, 0]);   % [1,0] = shift down by 1 row = vertical gradient
```
`circshift(S, [1,0])` shifts S down by 1 row with circular boundary. Subtracting gives the first-order finite difference approximation to `grad_y S`.

---

#### Term 2: `lambda1 * ||grad_x(O - S)||_{wn,1}` — Horizontal Sparsity of Clean Image Gradient

**Math:**
```
grad_x(O - S)  =  (O(i,j) - O(i,j-1)) - (S(i,j) - S(i,j-1))

Weighted L1-norm:
||X||_{wn,1}  =  sum_i sum_j  wn_{i,j} * |X_{i,j}|
```

**Meaning:**
Stripe noise appears prominently in the horizontal gradient of the observed image O — it creates sharp horizontal transitions at stripe boundaries. The difference `(O - S)` should be the clean image D, whose horizontal gradient should be sparse (most pixel pairs are similar, only true edges cause large differences). The **weighted** version with `wn` adapts at each iteration — pixels where the current estimate looks more like stripe noise receive higher weights, making the algorithm focus on those locations.

**In MATLAB code:**
```matlab
gradx_O = O - circshift(O, [0, 1]);   % [0,1] = shift right by 1 col = horizontal gradient
gradx_S = S - circshift(S, [0, 1]);
```

---

#### Term 3: `lambda2 * ||S||_{wg,2,1}` — Group Sparsity of Stripe Noise Columns

**Math:**
```
Standard L2,1-norm:
||S||_{2,1}  =  sum_j  ||S_{[j]}||_2   =  sum_j  sqrt( sum_i S(i,j)^2 )

Weighted version:
||S||_{wg,2,1}  =  sum_j  wg_j * ||S_{[j]}||_2
```
where `S_{[j]}` is the j-th column of S (the j-th "group").

**Meaning:**
Stripe noise is column-sparse — only a fraction of columns (typically 20-60%) actually contain stripes, while most columns are zero. The L2,1-norm promotes this group sparsity by penalizing the L2-norm of each column as a group. Columns with no stripe are driven to exactly zero. The adaptive weight `wg_j` is lower for columns more likely to be stripe noise (making them easier to activate) and higher for clean columns (protecting them from accidental assignment).

**In MATLAB code:**
```matlab
% Subproblem C uses group-wise soft-thresholding:
eta = S + tau3 / rho3;
for j = 1:w
    norm_eta = norm(eta(:,j), 2);           % L2-norm of column j
    thresh   = wg(j) * lambda2 / rho3;     % Adaptive threshold for this column
    if norm_eta > thresh
        C(:,j) = eta(:,j) * (norm_eta - thresh) / norm_eta;  % Block soft-threshold
    end
    % If norm_eta <= thresh: C(:,j) = 0 (this column has no stripe)
end
```

---

### Constrained Objective Function

To solve the objective function efficiently using ADMM, three **auxiliary variables** A, B, C are introduced and the problem is converted to a constrained form:

```
argmin_{A,B,C,S}  { ||A||_1  +  lambda1 * ||B||_{wn,1}  +  lambda2 * ||C||_{wg,2,1} }

subject to:
    A = grad_y S          (A captures the vertical gradient of S)
    B = grad_x(O - S)     (B captures the horizontal gradient of the clean image)
    C = S                 (C is an auxiliary copy of S for the group sparsity term)
```

**Why do this?**
The original objective mixes S and its gradients in complex, non-separable ways. By splitting them into separate variables with equality constraints, each variable can be updated independently with a closed-form solution, while the constraints enforce consistency. This is the core idea of ADMM.

---

### Augmented Lagrangian Function

The constraints are enforced by converting the constrained problem into an **augmented Lagrangian** (penalty + multiplier form):

```
L_rho(A, B, C, S, tau1, tau2, tau3)  =

    ||A||_1
    + tau1^T * (grad_y S - A)    +  (rho1/2) * ||grad_y S - A||_2^2

    + lambda1 * ||B||_{wn,1}
    + tau2^T * (grad_x O - grad_x S - B)  +  (rho2/2) * ||grad_x O - grad_x S - B||_2^2

    + lambda2 * ||C||_{wg,2,1}
    + tau3^T * (S - C)         +  (rho3/2) * ||S - C||_2^2
```

**Variables and their roles:**

| Symbol | Name | Role |
|--------|------|------|
| `tau1, tau2, tau3` | Lagrange multipliers | Dual variables enforcing equality constraints. Updated each iteration to push A toward grad_yS, B toward grad_x(O-S), C toward S |
| `rho1, rho2, rho3` | Penalty parameters | Control the strength of quadratic penalty for constraint violation. Larger rho = tighter enforcement |
| `(rho/2)||...||^2` | Quadratic penalty | Penalizes deviation from constraints and stabilizes ADMM iterations |

---

## Optimization Process — 4 Steps Per Iteration

The algorithm iterates the following 4 steps until convergence or `max_iter` is reached:

```
Initialization:
  S      = zeros(h, w)     <- Start with zero stripe estimate
  S_prev = S
  tau1 = tau2 = tau3 = 0   <- Lagrange multipliers start at zero
  alpha  = 1               <- Momentum coefficient
  wn     = 1               <- Norm weight (scalar)
  wg     = ones(1, w)      <- Group norm weights (one per column)
  k      = 0               <- Iteration counter

Loop: while k < max_iter
  Step 1: Weight Control           (update wn, wg using gamma and alpha)
  Step 2: Starting Point Control   (update alpha and damping d)
  Step 3: Step-Size Control        (update S and tau1/2/3 using momentum)
  Step 4: ADMM Subproblem Solving  (solve A -> B -> C -> S, then update tau)
  Convergence Check
```

---

### Step 1: Weight Control

**Purpose:** Adapt `wn` and `wg` at every iteration so the weighted norms dynamically focus on detecting stripe-like patterns vs. true image content.

#### 1a. Compute Residual Parameter gamma

```
gamma^k  =  ||(O - S^k) - (O - S^(k-1))||_F
             ----------------------------------
                  ||(O - S^(k-1))||_F
```

This is the relative change in the clean image estimate between iterations.

**In MATLAB code:**
```matlab
res_curr      = O - S;                           % Current clean estimate
res_prev      = O - S_prev;                      % Previous clean estimate
norm_res_prev = norm(res_prev, 'fro');           % Frobenius norm

if norm_res_prev == 0
    gamma = 0;                                   % Avoid division by zero
else
    gamma = norm(res_curr - res_prev, 'fro') / norm_res_prev;
end
```

- `norm(X, 'fro')` computes the Frobenius norm: `sqrt(sum of all squared elements)`
- `gamma` near 0 = solution is stable (near convergence)
- `gamma` large = solution is still changing significantly

---

#### 1b. Update Norm Weight wn

```
wn^(k+1)  =  (alpha^(k-1) + gamma^k)
              --------------------------
              (alpha^k - gamma^k)
```

`wn` is a scalar that modulates the threshold in Subproblem B. When gamma is large (rapid changes), wn increases, making the weighted L1 threshold more aggressive.

**In MATLAB code:**
```matlab
if alpha - gamma ~= 0
    wn = (alpha_prev + gamma) / (alpha - gamma);
else
    wn = 1;     % Safe fallback to avoid division by zero
end
```

---

#### 1c. Update Group Norm Weights wg

For each column j of S:

```
Step 1: Compute mean group norm:
    v^k  =  (1/n) * sum_j ||S^k_{[j]}||_2

Step 2: Check adjacency condition — update wg_j only if:
    ||S_{[j]} - S_{[j-1]}||_2 < v   AND   ||S_{[j]} - S_{[j+1]}||_2 < v
    (column j is similar to its neighbors = likely a smooth stripe, not an edge)

Step 3: If condition met:
    wg_j^(k+1)  =  1 / (2 * ||S^k_{[j]} + gamma^k||_2)
```

`wg_j` is **inversely proportional** to the current column norm. Strong stripe candidates (large column norm) get a small weight, making them easier to threshold. Clean columns (small norm) get large weights, protecting them.

**In MATLAB code:**
```matlab
column_norms = sqrt(sum(S.^2, 1));   % L2-norm of each column: 1 x w vector
v = sum(column_norms) / w;           % Mean column norm

for j = 1:w
    update_flag = true;
    % Check left neighbor
    if j > 1 && norm(S(:,j) - S(:,j-1), 2) >= v
        update_flag = false;         % Too different from neighbor: likely an edge
    end
    % Check right neighbor
    if j < w && norm(S(:,j) - S(:,j+1), 2) >= v
        update_flag = false;
    end
    if update_flag
        norm_shift = norm(S(:,j) + gamma, 2);   % ||S_j + gamma||_2
        if norm_shift > 0
            wg(j) = 1 / (2 * norm_shift);       % Inverse weighting
        end
    end
end
```

---

### Step 2: Evidence-Based Starting Point Control

**Purpose:** Update the momentum coefficient alpha and damping coefficient d to control how aggressively the algorithm accelerates. The threshold parameter `p` acts as a switch between two regimes.

#### 2a. Update Momentum Coefficient alpha (Nesterov's Method)

```
alpha^(k+1)  =  (1 + sqrt(4*(alpha^k)^2 + 1)) / 2,   if k <= p   <- aggressive phase
                (1 + sqrt(2*(alpha^k)^2 + 1)) / 2,   if k >  p   <- conservative phase
```

- When `k <= p`: uses `4*(alpha^k)^2` — more aggressive acceleration
- When `k > p`: uses `2*(alpha^k)^2` — gentler acceleration to avoid overshooting near convergence
- This is a variant of **Nesterov's accelerated gradient method** adapted for ADMM

**In MATLAB code:**
```matlab
alpha_prev = alpha;
if k <= p
    alpha = (1 + sqrt(1 + 4 * alpha_prev^2)) / 2;   % Aggressive Nesterov step
else
    alpha = (1 + sqrt(1 + 2 * alpha_prev^2)) / 2;   % Conservative Nesterov step
end
```

#### 2b. Update Damping Coefficient d

```
d^(k+1)  =  wn^(k+1),          if k <= p   <- Use norm weight for early damping
             alpha^k / alpha^(k+1),  if k >  p   <- Use ratio of momentum coefficients
```

`d` scales the Lagrange multipliers in Step 3, preventing excessive acceleration from destabilizing the optimization.

**In MATLAB code:**
```matlab
if k <= p
    d = wn;                      % Early phase: tie damping to norm weight
else
    d = alpha_prev / alpha;      % Late phase: standard Nesterov damping ratio
end
```

---

### Step 3: Momentum-Based Step-Size Control

**Purpose:** Apply a momentum-based extrapolation to S and scale Lagrange multipliers to accelerate convergence.

#### 3a. Momentum Update of S

```
S^k  <-  S^k  +  ((alpha^(k+1) - delta) / alpha^k)  *  (S^k - S^(k-1))
```

This is the **momentum extrapolation step**: instead of using just the current S, we overshoot in the direction of recent progress. `delta = 0.1` is a small damping constant that slightly reduces the step to prevent instability.

**In MATLAB code:**
```matlab
delta = 0.1;    % Damping constant from the paper
if alpha_prev ~= 0
    S = S + ((alpha - delta) / alpha_prev) * (S - S_prev);
end
```

#### 3b. Scale Lagrange Multipliers

```
tau1^k <- d^(k+1) * tau1^k
tau2^k <- d^(k+1) * tau2^k
tau3^k <- d^(k+1) * tau3^k
```

Scaling the multipliers by d prevents them from growing too large. This gives ADOM its stability advantage over plain accelerated ADMM.

**In MATLAB code:**
```matlab
tau1 = d * tau1;
tau2 = d * tau2;
tau3 = d * tau3;
```

---

### Step 4: ADMM-Based Subproblem Solving

This step solves four subproblems in sequence: A, B, C, then S. Each has a closed-form solution derived from the augmented Lagrangian.

---

#### Subproblem A — Pixel-wise Soft-Thresholding

**Minimize:** `||A||_1 + (rho1/2) * ||grad_y S - A + tau1/rho1||_2^2`

**Closed-form solution (soft-thresholding):**
```
A^(k+1)  =  sign(grad_y S^k + tau1^k/rho1)
             * max(|grad_y S^k + tau1^k/rho1| - 1/rho1, 0)
```

This is the **proximal operator of the L1-norm** (soft-thresholding). For each pixel independently:
- If `|x| > threshold`: shrink toward zero by the threshold amount
- If `|x| <= threshold`: set to exactly zero (promotes sparsity)

**In MATLAB code:**
```matlab
grady_S = S - circshift(S, [1, 0]);         % Vertical gradient of current S
temp = grady_S + tau1 / rho1;               % Shifted gradient
A = sign(temp) .* max(abs(temp) - 1/rho1, 0);  % Soft-threshold
```

`sign(temp)` gives the element-wise sign (+/-1), `max(abs(temp) - threshold, 0)` performs shrinkage, and `.*` is element-wise multiplication.

---

#### Subproblem B — Weighted Pixel-wise Soft-Thresholding

**Minimize:** `lambda1 * ||B||_{wn,1} + (rho2/2) * ||grad_x O - grad_x S - B + tau2/rho2||_2^2`

**Closed-form solution:**
```
B^(k+1)  =  sign(grad_x O - grad_x S^k + tau2^k/rho2)
             * max(|grad_x O - grad_x S^k + tau2^k/rho2| - wn^(k+1)*lambda1/rho2, 0)
```

Same structure as Subproblem A, but:
- The input signal is the horizontal gradient of the residual `grad_x O - grad_x S`
- The threshold is now `wn*lambda1/rho2` — adaptively scaled by the norm weight `wn`

**In MATLAB code:**
```matlab
gradx_O = O - circshift(O, [0, 1]);         % Horizontal gradient of O
gradx_S = S - circshift(S, [0, 1]);         % Horizontal gradient of current S
temp = gradx_O - gradx_S + tau2 / rho2;    % Shifted horizontal gradient of residual
B = sign(temp) .* max(abs(temp) - (wn * lambda1 / rho2), 0);   % Weighted soft-threshold
```

---

#### Subproblem C — Group-wise Block Soft-Thresholding

**Minimize:** `lambda2 * ||C||_{wg,2,1} + (rho3/2) * ||S - C + tau3/rho3||_2^2`

**Closed-form solution (block soft-thresholding, per column j):**
```
eta_j  =  S^k_{[j]} + tau3^k_{[j]} / rho3        <- Shifted S for column j

C^(k+1)_{[j]}  =  max(||eta_j||_2 - wg_j*lambda2/rho3, 0)
                   -----------------------------------------  * eta_j
                              ||eta_j||_2
```

This is the **proximal operator of the weighted L2,1-norm** (group soft-thresholding). For each column j:
- If the column's L2-norm exceeds the threshold: shrink the entire column proportionally toward zero
- If the norm is below the threshold: set the entire column to zero (no stripe in this column)

The direction of eta_j is preserved; only its magnitude is shrunk.

**In MATLAB code:**
```matlab
eta = S + tau3 / rho3;      % eta = S + tau3/rho3 (shifted S)
C   = zeros(h, w);          % Most columns will stay zero

for j = 1:w
    norm_eta = norm(eta(:,j), 2);              % L2-norm of column j of eta
    thresh   = wg(j) * lambda2 / rho3;        % Adaptive threshold for column j
    if norm_eta > thresh
        C(:,j) = eta(:,j) * (norm_eta - thresh) / norm_eta;   % Block soft-threshold
    end
end
```

---

#### Subproblem S — FFT-Based Least Squares Solver

**Minimize (jointly over S):**
```
(rho1/2) * ||grad_y S - A + tau1/rho1||_2^2
+ (rho2/2) * ||grad_x O - grad_x S - B + tau2/rho2||_2^2
+ (rho3/2) * ||S - C + tau3/rho3||_2^2
```

This is a **quadratic least squares problem** in S. The gradient set to zero gives:

```
(rho1 * grad_y^T grad_y  +  rho2 * grad_x^T grad_x  +  rho3 * I) * S  =  RHS

RHS  =  rho1 * grad_y^T (A^(k+1) - tau1^k/rho1)
       + rho2 * grad_x^T (grad_x O - B^(k+1) + tau2^k/rho2)
       + rho3 * (C^(k+1) - tau3^k/rho3)
```

Under **periodic boundary conditions**, finite-difference operators are circulant matrices, and the system can be solved efficiently using the **2D Fast Fourier Transform (FFT)**:

```
S^(k+1)  =  IFFT2( P / Q )

P  =  rho1 * FFT2(grad_y)* (element-wise) FFT2(A^(k+1) - tau1/rho1)
     + rho2 * FFT2(grad_x)* (element-wise) FFT2(grad_x O - B^(k+1) + tau2/rho2)
     + rho3 * FFT2(C^(k+1) - tau3/rho3)

Q  =  rho1 * FFT2(grad_y)* (element-wise) FFT2(grad_y)
     + rho2 * FFT2(grad_x)* (element-wise) FFT2(grad_x)
     + rho3
```

Where `*` denotes complex conjugate and all multiplications are element-wise.

**Key efficiency note:** `Q` only depends on fixed penalty parameters and image dimensions — it is **precomputed once before the loop** and reused every iteration.

**In MATLAB code:**
```matlab
%% Precomputed ONCE before the main loop:
[Fx, Fy] = meshgrid(0:w-1, 0:h-1);         % Frequency coordinate grids
F_x = 1 - exp(-1i * 2 * pi * Fx / w);      % DFT of horizontal difference operator
F_y = 1 - exp(-1i * 2 * pi * Fy / h);      % DFT of vertical difference operator
F_x_conj = conj(F_x);
F_y_conj = conj(F_y);

Q = rho1*(F_y_conj .* F_y) + rho2*(F_x_conj .* F_x) + rho3;
Q(Q == 0) = eps;     % Avoid division by zero at DC frequency

%% Inside the main loop — compute P and solve:
rhs1 = A - tau1 / rho1;                         % A^(k+1) - tau1/rho1
rhs2 = gradx_O - B + tau2 / rho2;              % grad_x O - B^(k+1) + tau2/rho2
rhs3 = C - tau3 / rho3;                         % C^(k+1) - tau3/rho3

P = rho1 * F_y_conj .* fft2(rhs1) ...
  + rho2 * F_x_conj .* fft2(rhs2) ...
  + rho3 * fft2(rhs3);

S_new = real(ifft2(P ./ Q));    % Take real part to eliminate floating-point imaginary residuals
```

---

#### Update Lagrange Multipliers

After all subproblems are solved, the Lagrange multipliers are updated using the **dual ascent step**:

```
tau1^(k+1)  =  tau1^k  +  rho1 * (grad_y S^(k+1) - A^(k+1))
tau2^(k+1)  =  tau2^k  +  rho2 * (grad_x O - grad_x S^(k+1) - B^(k+1))
tau3^(k+1)  =  tau3^k  +  rho3 * (S^(k+1) - C^(k+1))
```

Each update accumulates the **constraint violation** scaled by rho. If the constraint is exactly satisfied, the multiplier does not change. As long as there is residual violation, tau pushes the solution toward satisfying the constraint.

**In MATLAB code:**
```matlab
grady_Snew = S_new - circshift(S_new, [1, 0]);
gradx_Snew = S_new - circshift(S_new, [0, 1]);

tau1 = tau1 + rho1 * (grady_Snew - A);
tau2 = tau2 + rho2 * (gradx_O - gradx_Snew - B);
tau3 = tau3 + rho3 * (S_new - C);

S_prev = S;      % Save for next iteration's momentum computation
S = S_new;       % Accept the new stripe estimate
```

---

### Convergence Check

The iteration terminates when the relative change in the clean image estimate falls below tolerance:

```
||(O - S^(k+1)) - (O - S^k)||_F
---------------------------------  <=  tol
      ||(O - S^k)||_F
```

**In MATLAB code:**
```matlab
res_curr      = O - S;
res_prev      = O - S_prev;
norm_res_prev = norm(res_prev, 'fro');

if norm_res_prev > 0 && norm(res_curr - res_prev, 'fro') / norm_res_prev <= tol
    break;    % Converged: exit the loop early
end
```

After convergence (or reaching `max_iter`):
```matlab
destriped = O - S;    % Recover clean image by subtracting the estimated stripe noise
```

---

## Code File Explanations

---

### 1. `ADOM_vert.mlx` — Vertical Stripe Removal

**What it does:** Removes vertical stripe noise (stripes running top-to-bottom) from a grayscale image.

**Core function signature:**
```matlab
function destriped = ADOM(O, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter)
```

**Key design choices for vertical stripes:**
- Along-stripe direction = vertical (`grad_y`): stripes are constant down columns
- Perpendicular direction = horizontal (`grad_x`): stripes are sharp across rows
- Groups = columns (each column is one potential stripe)
- `wg` is a `1 x w` vector, one weight per column

**Initialization block:**
```matlab
[h, w] = size(O);
delta  = 0.1;               % Damping constant delta from the paper

S      = zeros(h, w);       % Start with no stripe estimate
S_prev = S;                 % Previous S for momentum tracking
tau1   = zeros(h, w);       % Lagrange multipliers, all zero at start
tau2   = zeros(h, w);
tau3   = zeros(h, w);
alpha  = 1;                 % Momentum coefficient starts at 1
wn     = 1;                 % Norm weight starts at neutral (1 = no weighting)
wg     = ones(1, w);        % Group norm weights, all 1 = equal weighting per column
k      = 0;                 % Iteration counter
```

**Synthetic vertical stripe generation (demo section):**
```matlab
stripe_ratio    = 0.4;      % 40% of columns get stripes
noise_intensity = 0.5;      % Stripe amplitude: values in range [-0.5, 0.5]
num_stripes     = round(stripe_ratio * size(O,2));
stripe_cols     = randperm(size(O,2), num_stripes);  % Randomly pick columns

for col = stripe_cols
    stripes(:, col) = randn() * noise_intensity;  % Same Gaussian value for ALL rows of this column
end
```

`randn()` produces a single Gaussian-distributed random number. Using the same value for all rows of a column creates a perfect vertical stripe.

---

### 2. `ADOM_hori.mlx` — Horizontal Stripe Removal

**What it does:** Removes horizontal stripe noise (stripes running left-to-right) from a grayscale image.

**Core function:**
```matlab
function destriped = ADOM_Horizontal(O, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter)
```

**Key differences from vertical version:**

| Aspect | Vertical (ADOM_vert) | Horizontal (ADOM_hori) |
|--------|----------------------|------------------------|
| Along-stripe gradient | `grad_y` (vertical) | `grad_x` (horizontal) |
| Perpendicular gradient | `grad_x` (horizontal) | `grad_y` (vertical) |
| Groups | Columns (each h x 1) | Rows (each 1 x w) |
| wg size | `1 x w` | `1 x h` |
| Subproblem A | Uses `grad_y S` | Uses `grad_x S` |
| Subproblem B | Uses `grad_x(O-S)` | Uses `grad_y(O-S)` |

**The FFT denominator Q is swapped to match the new gradient directions:**
```matlab
% Vertical version:
Q = rho1*(F_y_conj .* F_y) + rho2*(F_x_conj .* F_x) + rho3;

% Horizontal version (rho1 and rho2 operators swapped):
Q = rho1*(F_x_conj .* F_x) + rho2*(F_y_conj .* F_y) + rho3;
```

**Group norm weight update operates on rows instead of columns:**
```matlab
row_norms = sqrt(sum(S.^2, 2));   % sum along dim 2 (columns) -> L2-norm of each row: h x 1
v = sum(row_norms) / h;           % Mean row norm

for i = 1:h
    update_flag = true;
    if i > 1 && norm(S(i,:) - S(i-1,:), 2) >= v, update_flag = false; end  % Check row above
    if i < h && norm(S(i,:) - S(i+1,:), 2) >= v, update_flag = false; end  % Check row below
    if update_flag
        norm_shift = norm(S(i,:) + gamma, 2);
        if norm_shift > 0
            wg(i) = 1 / (2 * norm_shift);
        end
    end
end
```

**Subproblem A uses horizontal gradient (along-stripe direction for horizontal stripes):**
```matlab
gradx_S = S - circshift(S, [0, 1]);     % [0,1] = horizontal shift
temp = gradx_S + tau1 / rho1;
A = sign(temp) .* max(abs(temp) - 1/rho1, 0);
```

**Subproblem B uses vertical gradient (perpendicular direction for horizontal stripes):**
```matlab
grady_O = O - circshift(O, [1, 0]);     % Vertical gradient of O
grady_S = S - circshift(S, [1, 0]);
temp = grady_O - grady_S + tau2 / rho2;
B = sign(temp) .* max(abs(temp) - (wn * lambda1 / rho2), 0);
```

**Subproblem C operates on rows instead of columns:**
```matlab
for i = 1:h
    norm_eta = norm(eta(i,:), 2);             % L2-norm of row i
    thresh   = wg(i) * lambda2 / rho3;
    if norm_eta > thresh
        C(i,:) = eta(i,:) * (norm_eta - thresh) / norm_eta;
    end
end
```

---

### 3. `ADOM_diag.mlx` — Diagonal Stripe Removal

**What it does:** Removes diagonal stripe noise that runs along the main diagonal direction (top-left to bottom-right).

**Core function:**
```matlab
function destriped = ADOM(O, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter)
```

**Key challenge:** Diagonal stripes do not align with rows or columns, so groups are **diagonal index sets** that must be precomputed.

**Diagonal index precomputation:**
```matlab
ks = -(w-1):(h-1);          % Diagonal offsets: -(w-1) to (h-1), covers all diagonals
num_diags = length(ks);      % Total diagonals = h + w - 1

idxs = cell(1, num_diags);   % Cell array storing linear indices for each diagonal
for g = 1:num_diags
    k = ks(g);
    rows = max(1, k+1):min(h, k+w);    % Row range where diagonal g exists
    cols = rows - k;                   % Corresponding columns: col = row - k
    idxs{g} = sub2ind([h, w], rows, cols);  % Convert (row,col) to linear indices
end
```

`sub2ind([h,w], rows, cols)` converts 2D subscripts to linear (column-major) indices so diagonal elements can be addressed as a single vector.

**Diagonal gradient operators:**
```matlab
% Along diagonal: shift by [1,1] = down 1 row AND right 1 col
grad_diag_S = S - circshift(S, [1, 1]);

% Perpendicular (anti-diagonal): shift by [1,-1] = down 1 row AND left 1 col
grad_perp_S = S - circshift(S, [1, -1]);
grad_perp_O = O - circshift(O, [1, -1]);
```

**Diagonal Fourier multipliers:**
```matlab
% DFT of diagonal operator [1,1]:
F_diag = 1 - exp(-1i * 2 * pi * (Fx/w + Fy/h));

% DFT of anti-diagonal operator [1,-1]:
F_perp = 1 - exp(-1i * 2 * pi * (Fy/h - Fx/w));

Q = rho1*(conj(F_diag).*F_diag) + rho2*(conj(F_perp).*F_perp) + rho3;
Q(Q==0) = eps;
```

**Subproblem C for diagonals (reads and writes diagonal groups using precomputed indices):**
```matlab
eta = S + tau3 / rho3;
C   = zeros(h, w);
for g = 1:num_diags
    eta_group = eta(idxs{g});             % Extract diagonal g as a vector
    norm_eta  = norm(eta_group, 2);
    thresh    = wg(g) * lambda2 / rho3;
    if norm_eta > thresh
        c_group    = eta_group * (norm_eta - thresh) / norm_eta;
        C(idxs{g}) = c_group;             % Write back to diagonal positions
    end
end
```

---

### 4. `ADOM_2D.mlx` — Bidirectional Stripe Removal

**What it does:** Removes both vertical AND horizontal stripe noise by running two sequential passes. Both `ADOM_Vertical` and `ADOM_Horizontal` functions are defined in the same file.

**Two-pass pipeline:**
```matlab
% Pass 1: Remove vertical stripes using parameters A1, B1
destriped_vert = ADOM_Vertical(O_striped, A1, B1, rho1, rho2, rho3, p, tol, max_iter);

% Pass 2: Remove horizontal stripes from the already-destriped image, using A2, B2
destriped = ADOM_Horizontal(destriped_vert, A2, B2, rho1, rho2, rho3, p, tol, max_iter);
```

Separate `lambda1`/`lambda2` values (`A1/B1` vs `A2/B2`) allow independent tuning for each pass.

**Bidirectional synthetic noise generation:**
```matlab
% 40% vertical stripes with uniform amplitude in [-0.5, 0.5]
for col = randperm(size(O,2), round(0.4 * size(O,2)))
    stripes_vert(:, col) = (rand()*2 - 1) * noise_intensity_vert;
end

% 40% horizontal stripes with uniform amplitude in [-0.5, 0.5]
for row = randperm(size(O,1), round(0.4 * size(O,1)))
    stripes_horz(row, :) = (rand()*2 - 1) * noise_intensity_horz;
end

O_striped = max(0, min(1, O + stripes_vert + stripes_horz));  % Clip to [0,1]
```

---

### 5. `ADOM_stripe.mlx` — Synthetic Stripe Noise Generator

**What it does:** A standalone utility that adds synthetic vertical stripe noise to a clean image and saves the result. Used to create test inputs for the destriping scripts.

**Full pipeline:**
```matlab
%% Step 1: Load and convert image
img = imread(img_path);
if size(img,3) > 1
    img = rgb2gray(img);      % Convert RGB to grayscale
end
O = im2double(img);           % Convert uint8 [0,255] to double [0.0, 1.0]

%% Step 2: Generate stripe pattern
stripe_ratio    = 0.4;        % 40% of columns get stripes
noise_intensity = 0.5;        % Amplitude: stripes have values near [-0.5, +0.5]
num_stripes     = round(stripe_ratio * size(O,2));
stripe_cols     = randperm(size(O,2), num_stripes);

stripes = zeros(size(O));
for col = stripe_cols
    stripes(:, col) = randn() * noise_intensity;  % Same Gaussian value for all rows
end

%% Step 3: Add stripes and clip to valid pixel range
O_striped = O + stripes;
O_striped = max(0, min(1, O_striped));   % Clamp to [0,1]

%% Step 4: Save striped image
imwrite(O_striped, output_path);
```

---

### 6. `ADOM_striperemoval.mlx` — Load and Remove Stripe Noise

**What it does:** Loads a pre-saved striped image (output of `ADOM_stripe.mlx` or a real noisy RSI), runs ADOM to remove vertical stripes, and saves the clean result.

**Full pipeline:**
```matlab
%% Load the striped image
img = imread(striped_img_path);
if size(img,3) > 1, img = rgb2gray(img); end
O_striped = im2double(img);

%% Set parameters
lambda1 = 0.05;   rho1 = 1;   p = 10;
lambda2 = 0.1;    rho2 = 1;   tol = 1e-4;
                  rho3 = 1;   max_iter = 200;

%% Run ADOM
destriped = ADOM(O_striped, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter);

%% Save result
imwrite(destriped, output_path);
```

This is the recommended entry point for **processing real remote sensing images** with existing stripe noise — no synthetic noise generation needed.

---

### 7. `All-destripe.mlx` — Unified Multi-Direction Removal

**What it does:** Provides a single unified `ADOM` function that handles all three directions via a `direction` parameter, plus a demo that adds and removes all three stripe types simultaneously.

**Unified function signature:**
```matlab
function destriped = ADOM(O, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, direction)
% direction: 'vertical' | 'horizontal' | 'diagonal'
```

**Direction-aware configuration using a switch-case and MATLAB function handles:**
```matlab
switch lower(direction)
    case 'vertical'
        shift_along = [1, 0];     % Along-stripe gradient direction
        shift_perp  = [0, 1];     % Perpendicular gradient direction
        num_groups  = w;
        get_group = @(S, g) S(:, g);                           % Extract column g
        set_group = @(C, v, g) assign_column(C, v, g);        % Set column g

    case 'horizontal'
        shift_along = [0, 1];
        shift_perp  = [1, 0];
        num_groups  = h;
        get_group = @(S, g) S(g, :);                           % Extract row g
        set_group = @(C, v, g) assign_row(C, v, g);

    case 'diagonal'
        shift_along = [1,  1];    % Main diagonal
        shift_perp  = [1, -1];    % Anti-diagonal
        num_groups  = length(ks);
        get_group = @(S, g) S(idxs{g});                       % Extract diagonal g
        set_group = @(C, v, g) assign_diagonal(C, v, idxs{g});
end
```

The use of **function handles** (`@(S,g) S(:,g)`) makes the main ADMM loop completely generic — the same code for Subproblems A, B, C, and S works identically for all three directions without any direction-specific branching inside the loop.

**Generic Fourier multipliers computed directly from shift vectors:**
```matlab
exp_along    = exp(-1i * 2*pi * (shift_along(1)*Fy/h + shift_along(2)*Fx/w));
F_along      = 1 - exp_along;
F_along_conj = conj(F_along);

exp_perp     = exp(-1i * 2*pi * (shift_perp(1)*Fy/h + shift_perp(2)*Fx/w));
F_perp       = 1 - exp_perp;
F_perp_conj  = conj(F_perp);

Q = rho1*(F_along_conj .* F_along) + rho2*(F_perp_conj .* F_perp) + rho3;
Q(Q==0) = eps;
```

**Three-pass sequential destriping:**
```matlab
destriped_v = ADOM(O_striped,   ..., 'vertical');    % Pass 1: remove vertical
destriped_h = ADOM(destriped_v, ..., 'horizontal');  % Pass 2: remove horizontal
destriped   = ADOM(destriped_h, ..., 'diagonal');    % Pass 3: remove diagonal
```

**Three-direction synthetic noise generation:**
```matlab
noise_intensity = 0.3;   % Reduced intensity so all three types remain visible

% Vertical stripes
for col = randperm(w, round(stripe_ratio*w))
    stripes_v(:, col) = (rand()*2 - 1) * noise_intensity;
end

% Horizontal stripes
for row = randperm(h, round(stripe_ratio*h))
    stripes_h(row, :) = (rand()*2 - 1) * noise_intensity;
end

% Diagonal stripes (using precomputed diagonal index sets)
for g = randperm(num_diags, round(stripe_ratio*num_diags))
    stripes_d(idxs{g}) = (rand()*2 - 1) * noise_intensity;
end

O_striped = max(0, min(1, O + stripes_v + stripes_h + stripes_d));
```

---

## Repository Structure

```
├── ADOM_vert.mlx          # Vertical stripe removal (standalone ADOM function + demo)
├── ADOM_hori.mlx          # Horizontal stripe removal (ADOM_Horizontal function + demo)
├── ADOM_diag.mlx          # Diagonal stripe removal (diagonal ADOM function + demo)
├── ADOM_2D.mlx            # Bidirectional removal (two-pass: vertical then horizontal)
├── ADOM_stripe.mlx        # Utility: add synthetic vertical stripes to an image + save
├── ADOM_striperemoval.mlx # Utility: load a pre-striped image, run ADOM, save result
├── All-destripe.mlx       # Unified ADOM with direction param + three-pass all-direction demo
└── README.md
```

---

## Requirements

- **MATLAB** R2019b or later (Live Script `.mlx` format)
- **Image Processing Toolbox** — for `imread`, `imshow`, `rgb2gray`, `im2double`, `imwrite`
- No additional third-party toolboxes required

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Update image paths

In each `.mlx` file, update the hardcoded image path to your own image:

```matlab
img = imread("path/to/your/Image.jpg");
```

### 3. Run a script

Open any `.mlx` file in MATLAB and press **Run**. Each file is fully self-contained with the ADOM function definition and a demo section.

### 4. Suggested workflow for real remote sensing images

```
Real RSI with stripe noise
         |
         v
   ADOM_vert.mlx        <- If vertical stripes present
         |
         v
   ADOM_hori.mlx        <- If horizontal stripes also present
         |
         v
   ADOM_diag.mlx        <- If diagonal stripes also present
         |
         v
      Clean RSI
```

Or use `All-destripe.mlx` for a single-script solution handling all three directions automatically.

---

## Parameters Reference

| Parameter | Description | Recommended | Effect of Increasing |
|-----------|-------------|-------------|----------------------|
| `lambda1` | Weight for perpendicular gradient regularization | `0.01`–`0.1` | Stronger smoothing of image texture |
| `lambda2` | Weight for group sparsity regularization | `0.01`–`0.1` | More columns/rows/diagonals zeroed out |
| `rho1` | ADMM penalty for Subproblem A | `1` | Tighter constraint on along-stripe gradient |
| `rho2` | ADMM penalty for Subproblem B | `1` | Tighter constraint on perpendicular gradient |
| `rho3` | ADMM penalty for Subproblem C | `1` | Tighter constraint on group sparsity |
| `p` | Nesterov switch point (aggressive to conservative) | `10` | More aggressive acceleration in early iterations |
| `tol` | Convergence tolerance | `1e-4` | Smaller = more precise, more iterations needed |
| `max_iter` | Maximum iterations | `200` | Hard upper bound on computation time |

**Recommended lambda values by noise type (from paper experiments):**

| Noise Type | lambda1 | lambda2 |
|------------|---------|---------|
| Periodical stripes | `0.01` | `0.01` |
| Non-Periodical stripes | `0.01` | `0.01` |
| Broken stripes | `0.1` | `0.1` |
| Multiplicative stripes | `0.01` | `0.01` |
| Mixed (non-periodical + broken + wide) | `0.1` | `0.1` |

---

## Computational Complexity

| Subproblem | Operation | Per-Iteration Cost |
|------------|-----------|-------------------|
| A | Pixel-wise soft-thresholding | O(mn) |
| B | Pixel-wise weighted soft-thresholding | O(mn) |
| C | Group-wise block soft-thresholding | O(mn) |
| S | 2D FFT + pointwise division + 2D IFFT | O(mn log mn) |
| **Total per iteration** | | **O(mn log mn)** |
| **Total overall** | k iterations | **O(k * mn log mn)** |

- `m, n` = image height and width
- `k` = number of iterations until convergence (at most `max_iter`)
- The FFT denominator `Q` is **precomputed once** before the main loop, saving 3 FFT calls per iteration — a significant efficiency gain for large images

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{kim2023adom,
  title   = {ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image},
  author  = {Kim, Namwon and Han, Seong-Soo and Jeong, Chang-Sung},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {106587--106606},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3319268}
}
```

---

## License

This project is for academic and educational use. Please refer to the original paper's license:  
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Acknowledgements

- Original ADOM algorithm by Namwon Kim, Seong-Soo Han, and Chang-Sung Jeong (Korea University / Kangwon National University)
- This MATLAB implementation extends the original vertical-stripe model to support horizontal, diagonal, and multi-direction stripe removal
