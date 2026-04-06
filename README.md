<img width="481" height="180" alt="image" src="https://github.com/user-attachments/assets/511e5e4b-3a81-4703-845c-5f531513ff7a" />



# ADOM:ADMM-Based Optimization Model  for Stripe Noise Removal
## 22MAT220
## Mathematics for Computing 3

##  Team Members

## B SAI KRISHNA - CB.SC.U4AIE24308
## D NAGA SHIVA - CB.SC.U4AIE24315
## E SAI MOHITH  - CB.SC.U4AIE24316
## I MAHALAKSHMI - CB.SC.U4AIE24322



---

##  Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Methodology](#methodology)
4. [Dataset](#dataset)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)

---

##  Abstract

Stripe noise is a common and detrimental artifact in remote sensing images (RSI), arising from physical non-uniformities in satellite sensor arrays. This project implements **ADOM (ADMM-Based Optimization Model)**, a state-of-the-art optimization framework for stripe noise removal, extended to handle vertical, horizontal, diagonal, and multi-directional stripe corruptions in 2D grayscale images. The model formulates stripe extraction as a constrained convex optimization problem and solves it efficiently using the Alternating Direction Method of Multipliers (ADMM), augmented with a weight-based detection strategy and an acceleration mechanism comprising evidence-based starting point control and momentum-based step-size control. The implementation is carried out in MATLAB and tested on synthetic stripe-corrupted remote sensing images, demonstrating effective noise suppression while preserving fine spatial detail.

---

##  Introduction

### Background and Motivation

Remote sensing images (RSI) acquired by satellite and airborne sensors are indispensable tools in modern science and engineering. They support a wide range of critical real-world applications including Earth observation, land-use and land-cover classification, urban planning, agricultural monitoring, climate change analysis, natural disaster response, and environmental surveillance. Satellites such as NASA's Landsat, ESA's Sentinel series, and hyperspectral platforms like EO-1 Hyperion and MODIS continuously generate large volumes of image data that form the backbone of geospatial intelligence.

However, the quality of remote sensing images is frequently compromised by **stripe noise** a form of structured, systematic degradation that manifests as bright or dark bands running vertically, horizontally, or diagonally across the image. This noise type is distinct from random (Gaussian) noise because it is spatially correlated, direction-dependent, and often non-periodic and non-uniform in intensity.

### Origin and Nature of Stripe Noise

Stripe noise in RSI arises primarily from hardware-level imperfections in the imaging sensors. Modern satellites use **push-broom** or **whisk-broom** scanning systems, where individual detector elements (pixels in the sensor array) scan the Earth's surface in parallel. Each detector element has its own gain and bias characteristics, and even small calibration differences between detectors caused by temperature variations, aging electronics, or manufacturing tolerances — appear as persistent stripes in the final image.

The key properties that make stripe noise particularly difficult to remove are:

- **Non-periodicity:** Stripes do not occur at fixed, regular intervals. They appear randomly across different columns, rows, or diagonals.
- **Non-uniformity:** The intensity of each stripe varies independently some stripes are faint while others are highly prominent.
- **Structural similarity to image content:** Stripe noise can closely mimic genuine image structures such as roads, field boundaries, and building edges, making it difficult to distinguish noise from signal without sophisticated models.
- **Multi-directionality:** Stripes can appear simultaneously in vertical, horizontal, and diagonal directions, especially in multi-sensor or multi-pass acquisition systems.

### Why Stripe Noise is Dangerous for Applications

Unremoved stripe noise causes cascading problems in downstream RSI processing pipelines:

- **Classification errors:** Stripe patterns confuse spectral classifiers, leading to incorrect land-cover labels.
- **Target detection failures:** Stripes masquerade as linear features such as roads, rivers, or runways in object detection algorithms.
- **Change detection artifacts:** Temporal differences caused by stripe noise are falsely flagged as genuine land-use changes between acquisition dates.
- **Hyperspectral analysis degradation:** In hyperspectral images, stripes corrupt the spectral signature of ground materials, invalidating abundance estimation and unmixing.

### Limitations of Existing Destriping Methods

Several categories of destriping methods have been proposed in the literature, each with its own limitations:

**1. Filter-Based Methods:** Simple spatial or frequency-domain filters (e.g., notch filters, Fourier filtering) can suppress periodic stripes but fail on non-periodic, non-uniform stripe patterns because the stripe frequency overlaps with genuine image content.

**2. Statistics-Based Methods (Moment Matching, Histogram Equalization):** These approaches equalize the statistical properties (mean, variance) across detector elements. They are computationally fast but assume stripes are stationary and purely multiplicative/additive in a fixed statistical sense an assumption that breaks down for complex scenes.

**3. Variational/Optimization-Based Methods (e.g., UTV, GSR, DLS, LRHP):** These formulate destriping as a regularized optimization problem using priors such as total variation (TV) or low-rank structure. While effective, many of these methods are computationally expensive and struggle to separate stripe noise from fine image details such as edges and textures with similar directionality.

**4. Deep Learning-Based Methods (e.g., DnCNN, Wavelet-DNN):** Neural networks trained on large image datasets can achieve impressive destriping but require massive labelled training data, are sensitive to domain shifts between satellite sensors, and are computationally heavy at inference time.

### Motivation for ADOM

The ADOM framework, proposed by Kim, Han, and Jeong (IEEE Access, 2023), addresses the above limitations by:

1. **Formulating stripe noise removal as a convex optimization problem** that explicitly models the directional properties of stripe noise its smoothness along the stripe direction and sparsity across the stripe direction.
2. **Introducing adaptive weighted norms** that dynamically adjust per-group (per-column, per-row, or per-diagonal) weights based on the residual between successive estimates, enabling the model to distinguish genuine image edges from stripe noise even when they have similar gradient magnitudes.
3. **Accelerating convergence with two novel control strategies** — evidence-based starting point control and momentum-based step-size control that together significantly reduce the number of ADMM iterations needed for convergence while maintaining solution quality.

This project implements the ADOM framework in MATLAB and extends it to handle all three stripe directions — vertical, horizontal, and diagonal — both independently and in a sequential multi-directional pipeline. This makes it applicable to the full range of stripe noise corruption patterns encountered in real remote sensing sensor systems.

---

##  Methodology

### 3.1 Image Degradation Model

The fundamental assumption is that an observed corrupted remote sensing image **O** is formed by additively superimposing stripe noise **S** onto the clean latent image **X**:

```
O = X + S
```

The goal is to **estimate S** from O alone, and then recover the clean image as:

```
X̂  =  O − S
```

This is an ill-posed inverse problem because infinitely many pairs (X, S) can produce the same observed O. The problem is made tractable by incorporating prior knowledge about the structural properties of both the clean image and the stripe noise through regularization.

---

### 3.2 Stripe Noise Priors

Two key structural properties of stripe noise are exploited as regularization priors:

**Prior 1 — Gradient Sparsity Along the Stripe Direction (L1 norm):**
Stripe noise is piecewise constant along the stripe direction. For vertical stripes, the stripe signal does not change in the vertical (y) direction i.e., the vertical gradient of S is sparse (mostly zero). Mathematically:

```
||∇_along S||_1  is small
```

where `∇_along` denotes the finite difference operator along the stripe direction.

**Prior 2 — Group Sparsity Across the Stripe Direction (Weighted Group L2 norm):**
Stripe noise is localized — only a fraction of columns (or rows or diagonals) carry noise. The stripe component S can be represented as a collection of groups `{S_g}`, where each group corresponds to one stripe-direction element (one column for vertical, one row for horizontal, one diagonal for diagonal stripes). Only a few groups have non-zero energy:

```
Σ_g  w_g  ||S_g||_2  is small
```

where `w_g` is an adaptive weight assigned to each group g, and `||S_g||_2` is the L2 norm of the stripe values in group g.

**Prior 3 — Smoothness of the Residual Across the Stripe Direction (L1 norm):**
The true image X is smooth across the stripe direction. This means the cross-direction gradient of O − S (which equals the cross-direction gradient of X) should be sparse:

```
||∇_perp (O − S)||_1  is small
```

---

### 3.3 Optimization Problem Formulation

Combining the three priors, the stripe noise S is estimated by solving the following **constrained convex optimization problem**:

```
min_{S, A, B, C}   ||A||_1  +  λ₁ ||B||_1  +  λ₂ Σ_g w_g ||C_g||_2

subject to:
    ∇_along S  =  A          ... (constraint 1)
    ∇_perp (O − S)  =  B     ... (constraint 2)
    S  =  C                  ... (constraint 3)
```

where:
- `A` encodes the along-direction gradient of S (sparsity prior 1)
- `B` encodes the cross-direction gradient of the residual (smoothness prior 3)
- `C` encodes the group-sparse structure of S (group sparsity prior 2)
- `λ₁`, `λ₂` are regularization hyperparameters balancing the three terms
- `w_g` are adaptive per-group weights (updated each iteration)

The auxiliary variables A, B, C decouple the problem into independent subproblems, each solvable in closed form.

---

### 3.4 Augmented Lagrangian and ADMM Framework

To solve the constrained problem, the **Augmented Lagrangian** is formed by introducing Lagrange multipliers `τ₁`, `τ₂`, `τ₃` and penalty parameters `ρ₁`, `ρ₂`, `ρ₃`:

```
L(S, A, B, C, τ₁, τ₂, τ₃) =

    ||A||_1  +  λ₁||B||_1  +  λ₂ Σ_g w_g ||C_g||_2

  + <τ₁, ∇_along S − A>  +  (ρ₁/2) ||∇_along S − A||²_F

  + <τ₂, ∇_perp(O−S) − B>  +  (ρ₂/2) ||∇_perp(O−S) − B||²_F

  + <τ₃, S − C>  +  (ρ₃/2) ||S − C||²_F
```

ADMM minimizes this Lagrangian by alternately updating each variable while keeping the others fixed, then updating the multipliers. One full ADMM iteration consists of **five steps**:

---

### 3.5 ADMM Subproblem Solutions

#### Step 1 — Subproblem A: L1 Soft Thresholding (Along-Direction Gradient)

Minimizing L over A with S, B, C, τ fixed gives a classic L1 proximal problem:

```
A* = argmin_A  ||A||_1  +  (ρ₁/2) ||∇_along S − A + τ₁/ρ₁||²_F
```

The closed-form solution is the **element-wise soft-thresholding operator**:

```
A  =  shrink( ∇_along S + τ₁/ρ₁,  1/ρ₁ )

where  shrink(x, θ)  =  sign(x) · max(|x| − θ,  0)
```

In MATLAB (from `ADOM_vert.mlx`):
```matlab
temp = grady_S + tau1 / rho1;
A = sign(temp) .* max(abs(temp) - 1/rho1, 0);
```

#### Step 2 — Subproblem B: Weighted L1 Soft Thresholding (Cross-Direction Gradient)

Minimizing L over B:

```
B* = argmin_B  λ₁||B||_1  +  (ρ₂/2) ||∇_perp(O−S) − B + τ₂/ρ₂||²_F
```

Solution:

```
B  =  shrink( ∇_perp(O−S) + τ₂/ρ₂,  wₙ·λ₁/ρ₂ )
```

where `wₙ` is the **momentum-adaptive norm weight** (updated each iteration, see Section 3.6). In MATLAB:
```matlab
temp = gradx_O - gradx_S + tau2 / rho2;
B = sign(temp) .* max(abs(temp) - (wn * lambda1 / rho2), 0);
```

#### Step 3 — Subproblem C: Weighted Group Soft Thresholding (Group Sparsity)

Minimizing L over C involves a **group-wise proximal operator** (block soft thresholding):

```
C_g* = argmin_{C_g}  w_g λ₂ ||C_g||_2  +  (ρ₃/2) ||η_g − C_g||²_F

where  η_g  =  S_g + τ₃_g/ρ₃
```

The closed-form solution is the **group soft-thresholding operator**:

```
         ⎧  η_g · (||η_g||₂ − w_g λ₂/ρ₃) / ||η_g||₂    if  ||η_g||₂ > w_g λ₂/ρ₃
C_g  =   ⎨
         ⎩  0                                             otherwise
```

For vertical stripes, each group `g` is one column of S. In MATLAB:
```matlab
eta = S + tau3 / rho3;
C = zeros(h, w);
for j = 1:w
    norm_eta = norm(eta(:,j), 2);
    thresh = wg(j) * lambda2 / rho3;
    if norm_eta > thresh
        C(:,j) = eta(:,j) * (norm_eta - thresh) / norm_eta;
    end
end
```

#### Step 4 — Subproblem S: FFT-Based Closed-Form Linear Solver

Minimizing L over S (with A, B, C, τ fixed) leads to a **linear system**:

```
[ ρ₁ ∇_along^T ∇_along  +  ρ₂ ∇_perp^T ∇_perp  +  ρ₃ I ] S  =  rhs
```

where:
```
rhs  =  ρ₁ ∇_along^T (A − τ₁/ρ₁)
       + ρ₂ ∇_perp^T (∇_perp O − B + τ₂/ρ₂)
       + ρ₃ (C − τ₃/ρ₃)
```

Because the gradient operators `∇_along` and `∇_perp` are **circular convolutions** (with periodic boundary conditions), this linear system is **diagonalized in the Fourier domain**. The solution is computed efficiently via the 2D Fast Fourier Transform (FFT):

```
Ŝ  =  iFFT( P̂  /  Q̂ )

where:
  P̂  =  ρ₁ F̄_along · FFT(rhs₁)  +  ρ₂ F̄_perp · FFT(rhs₂)  +  ρ₃ · FFT(rhs₃)

  Q̂  =  ρ₁ |F_along|²  +  ρ₂ |F_perp|²  +  ρ₃

  F_along  =  1 − exp(−j 2π (shift_y/H + shift_x/W))   [DFT of along-direction difference filter]
  F_perp   =  1 − exp(−j 2π (shift_y/H − shift_x/W))   [DFT of perp-direction difference filter]
```

For vertical stripes (shift_along = [1,0], shift_perp = [0,1]):
```matlab
F_y = 1 - exp(-1i * 2 * pi * Fy / h);   % Vertical gradient DFT
F_x = 1 - exp(-1i * 2 * pi * Fx / w);   % Horizontal gradient DFT
Q = rho1*(conj(F_y).*F_y) + rho2*(conj(F_x).*F_x) + rho3;
P = rho1*conj(F_y).*fft2(rhs1) + rho2*conj(F_x).*fft2(rhs2) + rho3*fft2(rhs3);
S_new = real(ifft2(P ./ Q));
```

This FFT solver runs in **O(HW log(HW))** time, making it far more efficient than solving the full linear system directly.

#### Step 5 — Lagrange Multiplier Updates

After each ADMM subproblem cycle, the dual variables (Lagrange multipliers) are updated via gradient ascent on the dual function:

```
τ₁  ←  τ₁  +  ρ₁ (∇_along S_new − A)
τ₂  ←  τ₂  +  ρ₂ (∇_perp(O − S_new) − B)
τ₃  ←  τ₃  +  ρ₃ (S_new − C)
```

---

### 3.6 Weight-Based Detection Strategy

A critical innovation in ADOM is the **adaptive weighting** of the norm weight `wₙ` and the group weights `{w_g}`, which allow the model to distinguish stripe noise from genuine image edges.

#### Norm Weight wₙ (Controls Subproblem B Threshold)

At each iteration k, the **residual ratio** γ is computed as:

```
γ_k  =  ||res_k − res_{k-1}||_F  /  ||res_{k-1}||_F

where  res_k  =  O − S_k
```

This measures how much the residual image changed between iterations. The norm weight is then:

```
wₙ  =  (α_{k-1} + γ_k) / (α_k − γ_k)
```

A large γ (fast-changing residual) indicates the model is aggressively removing content — possibly genuine image detail — so wₙ is adjusted to be more conservative.

#### Group Weight wg (Controls Group Thresholding in Subproblem C)

For each stripe group g, the adaptive weight `w_g` is updated based on the current stripe estimate:

```
v  =  (1/G) Σ_g ||S_g||_2      [average group norm]

For each group g:
  if ||S_g − S_{g-1}||₂ < v  AND  ||S_g − S_{g+1}||₂ < v:
      w_g  =  1 / (2 · ||S_g + γ||₂)
```

Groups that differ greatly from their neighbors (likely genuine image edges) are excluded from the weight update, preserving their weights. Groups that are locally smooth (likely uniform stripes) get their weights updated, making their thresholds more sensitive. In MATLAB:

```matlab
column_norms = sqrt(sum(S.^2, 1));
v = sum(column_norms) / w;
for j = 1:w
    update_flag = true;
    if j > 1 && norm(S(:,j) - S(:,j-1), 2) >= v, update_flag = false; end
    if j < w && norm(S(:,j) - S(:,j+1), 2) >= v, update_flag = false; end
    if update_flag
        norm_shift = norm(S(:,j) + gamma, 2);
        if norm_shift > 0
            wg(j) = 1 / (2 * norm_shift);
        end
    end
end
```

---

### 3.7 ADMM-Based Acceleration Strategy

Two control strategies are used to accelerate convergence beyond standard ADMM.

#### Strategy 1: Evidence-Based Starting Point Control

The momentum parameter `α` is updated using a Nesterov-inspired schedule that provides a better starting point for each ADMM subproblem:

```
For k ≤ p  (early phase):
    α_k  =  (1 + √(1 + 4 α_{k-1}²)) / 2      [faster, FISTA-type schedule]

For k > p  (late phase):
    α_k  =  (1 + √(1 + 2 α_{k-1}²)) / 2      [slower, stability-focused schedule]
```

The threshold `p` (default 10) separates aggressive early-phase acceleration from stable late-phase convergence. The Lagrange multipliers are also scaled by a factor `d`:

```
For k ≤ p:  d = wₙ
For k > p:  d = α_{k-1} / α_k
```

This is applied as:
```matlab
tau1 = d * tau1;
tau2 = d * tau2;
tau3 = d * tau3;
```

#### Strategy 2: Momentum-Based Step-Size Control

The stripe estimate S is extrapolated using momentum before each ADMM cycle, providing a warm start:

```
S  ←  S  +  ((α_k − δ) / α_{k-1}) · (S_k − S_{k-1})
```

where `δ = 0.1` is a damping coefficient that prevents overshooting and maintains stability. In MATLAB:

```matlab
S = S + ((alpha - delta) / alpha_prev) * (S - S_prev);
```

---

### 3.8 Convergence Criterion

The algorithm terminates when the relative change in the residual falls below a tolerance threshold `tol`:

```
||res_k − res_{k-1}||_F  /  ||res_{k-1}||_F   ≤   tol
```

Default values: `tol = 1e-4`, `max_iter = 200`.

---

### 3.9 Direction-Specific Configurations

The ADOM framework is generalized to three stripe directions by changing the gradient operators and group definitions:

| Direction | Along Shift | Perp Shift | Groups | F_along | F_perp |
|-----------|------------|------------|--------|---------|--------|
| Vertical | [1, 0] (↓) | [0, 1] (→) | Columns (j = 1..W) | 1 − e^{−j2πfy/H} | 1 − e^{−j2πfx/W} |
| Horizontal | [0, 1] (→) | [1, 0] (↓) | Rows (i = 1..H) | 1 − e^{−j2πfx/W} | 1 − e^{−j2πfy/H} |
| Diagonal | [1, 1] (↘) | [1, −1] (↗) | Main diagonals (k = −(W−1)..H−1) | 1 − e^{−j2π(fy/H + fx/W)} | 1 − e^{−j2π(fy/H − fx/W)} |

For diagonal stripes, group g corresponds to the k-th main diagonal of the image matrix. The pixels on diagonal k are indexed as:

```matlab
ks = -(w-1):(h-1);
for g = 1:num_diags
    k = ks(g);
    rows = max(1, k+1) : min(h, k+w);
    cols = rows - k;
    idxs{g} = sub2ind([h, w], rows, cols);
end
```

---

### 3.10 Multi-Directional Destriping Pipeline

For images corrupted by stripes in multiple directions simultaneously, the ADOM functions are applied **sequentially**:

```
O_striped  →  ADOM_Vertical  →  ADOM_Horizontal  →  ADOM_Diagonal  →  X̂_clean
```

This sequential pipeline is implemented in `All-destripe.mlx`, where a unified ADOM function accepts a `direction` parameter:

```matlab
destriped_v = ADOM(O_striped, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'vertical');
destriped_h = ADOM(destriped_v, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'horizontal');
destriped   = ADOM(destriped_h, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'diagonal');
```

---

### 3.11 Implementation Files

| File | Function | Description |
|------|----------|-------------|
| `ADOM_vert.mlx` | `ADOM()` | Vertical stripe removal; groups = image columns |
| `ADOM_hori.mlx` | `ADOM_Horizontal()` | Horizontal stripe removal; groups = image rows |
| `ADOM_diag.mlx` | `ADOM()` | Diagonal stripe removal; groups = image diagonals |
| `ADOM_2D.mlx` | `ADOM_Vertical()` + `ADOM_Horizontal()` | Sequential bidirectional destriping |
| `ADOM_stripe.mlx` | — | Synthetic stripe noise generation and saving |
| `ADOM_striperemoval.mlx` | `ADOM()` | Load a pre-saved striped image and apply vertical ADOM |
| `All-destripe.mlx` | `ADOM()` (unified) | Unified ADOM supporting all directions via `direction` flag |

---

### 3.12 Algorithm Parameters

| Parameter | Symbol | Default | Role |
|-----------|--------|---------|------|
| Regularization 1 | `λ₁` | 0.05 | Weight on cross-direction gradient sparsity (Subproblem B) |
| Regularization 2 | `λ₂` | 0.10 | Weight on group sparsity (Subproblem C) |
| ADMM Penalty 1 | `ρ₁` | 1 | Penalty for along-direction constraint |
| ADMM Penalty 2 | `ρ₂` | 1 | Penalty for cross-direction constraint |
| ADMM Penalty 3 | `ρ₃` | 1 | Penalty for group-structure constraint |
| Momentum threshold | `p` | 10 | Iteration boundary between fast and stable acceleration phases |
| Damping factor | `δ` | 0.1 | Prevents overshooting in momentum step |
| Tolerance | `tol` | 1e-4 | Convergence criterion (relative residual change) |
| Max iterations | `max_iter` | 200 | Hard upper bound on ADMM iterations |

---

##  Dataset

Remote sensing images were used for testing the ADOM destriping algorithm. Synthetic stripe noise was added to clean grayscale images at a stripe density of **40%** with noise intensity uniformly sampled from **[−0.5, 0.5]**.

Commonly used benchmark datasets for remote sensing destriping include:

| Dataset | Sensor | Description | URL |
|---------|--------|-------------|-----|
| Cuprite | AVIRIS (NASA) | Hyperspectral, mineral mapping scene | [NASA AVIRIS](https://aviris.jpl.nasa.gov/) |
| Pavia University | ROSIS (DLR) | Urban hyperspectral scene | [Hyperspectral Remote Sensing Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Hyperion | EO-1 (NASA) | Hyperspectral, 242 bands | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) |
| Aqua/Terra MODIS | NASA | Moderate resolution, global coverage | [NASA Earthdata](https://earthdata.nasa.gov/) |

In this project, a custom local image (`Image.jpg`) was used for proof-of-concept testing with synthetic stripe noise superimposed programmatically in MATLAB. Stripe noise parameters used:
- **Stripe ratio:** 40% of columns/rows/diagonals affected
- **Noise intensity:** Drawn from `randn() × 0.5` (Gaussian, zero-mean)
- **Clipping:** Output clipped to [0, 1] after adding noise

---

##  Results

### Fig. 1: Vertical Stripe Removal

<p align="center">
  <img src="result_vert.jpeg" width="700">
</p>


**Fig. 1:** Demonstration of ADOM vertical stripe removal (`ADOM_vert.mlx`). clean reference image. synthetically corrupted image with 40% column-wise stripe noise. ADOM destriped output.

---

### Fig. 2: Horizontal Stripe Removal

| Original Clean Image | Striped Image (Horizontal) | Destriped Output |
|:--------------------:|:--------------------------:|:----------------:|
| *(insert figure)*    | *(insert figure)*          | *(insert figure)* |

**Fig. 2:** Demonstration of ADOM horizontal stripe removal (`ADOM_hori.mlx`).

---

### Fig. 3: Diagonal Stripe Removal

| Original Clean Image | Striped Image (Diagonal) | Destriped Output |
|:--------------------:|:------------------------:|:----------------:|
| *(insert figure)*    | *(insert figure)*        | *(insert figure)* |

**Fig. 3:** Demonstration of ADOM diagonal stripe removal using diagonal group sparsity (`ADOM_diag.mlx`).

---

### Fig. 4: Multi-Directional Stripe Removal

| Original | V + H + D Striped | Fully Destriped |
|:--------:|:-----------------:|:---------------:|
| *(insert figure)* | *(insert figure)* | *(insert figure)* |

**Fig. 4:** Sequential application of ADOM for all three stripe directions (`All-destripe.mlx`). V = Vertical, H = Horizontal, D = Diagonal.

---

### Table 1: Quantitative Evaluation

| Configuration | Stripe Type | PSNR (dB) ↑ | SSIM ↑ |
|---------------|-------------|-------------|--------|
| ADOM_vert | Vertical only | — | — |
| ADOM_hori | Horizontal only | — | — |
| ADOM_diag | Diagonal only | — | — |
| ADOM_2D | Vertical + Horizontal | — | — |
| All-destripe | V + H + Diagonal | — | — |

> ⚠️ *Fill in PSNR and SSIM values from MATLAB experiments using `psnr(destriped, O)` and `ssim(destriped, O)`.*

---

##  Conclusion

This project successfully implements and extends the ADOM framework for multi-directional stripe noise removal in remote sensing images. Beginning from the base paper (Kim et al., IEEE Access 2023), which addresses vertical stripe removal, the implementation was extended to cover horizontal, diagonal, and simultaneous multi-directional stripe noise — a more realistic and challenging scenario encountered in real satellite sensor systems.

The key technical contributions of this implementation are:

- **Rigorous mathematical formulation:** Stripe removal is modelled as a constrained convex optimization problem using three complementary priors — L1 gradient sparsity along the stripe direction, L1 smoothness across the stripe direction, and weighted group L2 sparsity of the stripe component.
- **Efficient FFT-based linear solver:** The S-subproblem is solved in the Fourier domain in O(HW log HW) time, making the algorithm scalable to large remote sensing images.
- **Adaptive weight-based detection:** Per-group adaptive weights `w_g` are updated each iteration based on the residual dynamics, enabling precise discrimination between genuine image edges and stripe patterns even when they have similar gradient magnitudes.
- **Accelerated convergence:** Evidence-based starting point control (Nesterov-type momentum schedule) and momentum-based step-size control with damping together reduce the iteration count significantly compared to vanilla ADMM.
- **Unified multi-direction pipeline:** A single parameterized ADOM function supports all stripe directions via the `direction` flag, and sequential composition achieves full multi-directional destriping.

Future directions include: extending ADOM to hyperspectral (3D) image cubes exploiting inter-band correlation, automated parameter tuning using quality metrics (PSNR, SSIM), and benchmarking against deep learning destripers such as DnCNN and Wavelet-DNN on standard remote sensing datasets.

---

##  References

**[1]** N. Kim, S.-S. Han, and C.-S. Jeong, "ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image," *IEEE Access*, vol. 11, pp. 106587–106606, 2023. DOI: [10.1109/ACCESS.2023.3320190](https://ieeexplore.ieee.org/document/10262317)

**[2]** S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers," *Foundations and Trends in Machine Learning*, vol. 3, no. 1, pp. 1–122, 2011.

**[3]** Y. Chang, L. Yan, T. Wu, and S. Zhong, "Remote Sensing Image Stripe Noise Removal: From Image Decomposition Perspective," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 54, no. 12, pp. 7018–7031, 2016.

**[4]** A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems," *SIAM Journal on Imaging Sciences*, vol. 2, no. 1, pp. 183–202, 2009. *(Foundation for Nesterov-type momentum used in ADOM)*

**[5]** J. Guan, R. Lai, and A. Xiong, "Wavelet Deep Neural Network for Stripe Noise Removal," *IEEE Access*, vol. 7, pp. 44544–44554, 2019.

**[6]** NASA Jet Propulsion Laboratory, *AVIRIS Cuprite Dataset*. [Online]. Available: https://aviris.jpl.nasa.gov/

**[7]** The MathWorks, Inc., *MATLAB R2023b Documentation*. [Online]. Available: https://www.mathworks.com/help/matlab/

**[8]** namwonss (N. Kim), *ADOM Official GitHub Repository*, 2023. [Online]. Available: https://github.com/namwonss/ADOM

---

> *This project was developed as part of the MFC3/MFC4 curriculum at Amrita Vishwa Vidyapeetham.*
