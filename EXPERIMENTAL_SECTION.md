# Experimental Section: Double-Line Offset Analysis of Cyclic Voltammograms

## Electrochemical Measurements

Cyclic voltammetry (CV) experiments were performed on nickel-based catalysts (NiC) in alkaline medium (1 M KOH) with urea (330 mM) at room temperature. Measurements were conducted using a three-electrode configuration with the sample as the working electrode. Potential scans were conducted at a scan rate of 10 mV/s over a potential window spanning the redox-active regions of the catalyst.

## Data Processing and Analysis

### Baseline Data Preparation

Raw current data from 1000 consecutive CV cycles were imported and organized into oxidation (forward scan) and reduction (backward scan) branches, each represented as a matrix where rows correspond to cycle numbers and columns correspond to applied potential values. Current values were converted to current density (mA/cm²) by normalizing to the electrode area (0.196 cm²).

All potentials were referenced to the reversible hydrogen electrode (RHE) using the relationship:
$$E_{\text{RHE}} = E_{\text{measured}} + 1.002 \text{ V}$$

### Contour Plot Generation

Two-dimensional heatmaps were generated to visualize current density evolution as a function of potential and cycle number. Data gaps corresponding to mechanical drift regions (cycles 235–290 and 350–400) were bridged using linear interpolation between adjacent cycles to ensure visual continuity. The contour plots were generated at multiple resolutions (150, 300, and 600 dpi) with and without axis labels to facilitate publication.

### Double-Line Offset Analysis

A key innovation in the present analysis is the simultaneous fitting of parallel lines to the oxidation and reduction branches of each CV cycle. This approach quantifies the potential-dependent difference between the forward and backward scans, capturing electrochemical hysteresis with a single metric per cycle.

#### Fitting Methodology

For each cycle, the oxidation and reduction current data within a predefined potential window were fit to the linear model:
$$j_{\text{ox}}(E) = mE + b_{\text{ox}}$$
$$j_{\text{red}}(E) = mE + b_{\text{red}}$$

where $j$ is current density (mA/cm²), $E$ is potential (V vs RHE), $m$ is the common slope (conductivity), and $b_{\text{ox}}$ and $b_{\text{red}}$ are the intercepts for the oxidation and reduction branches, respectively.

The parallel-line constraint (common slope $m$) was enforced using weighted least-squares regression. The objective function minimized was:
$$\sum_{i} w_{i,\text{ox}}(j_{\text{ox},i} - mE_{\text{ox},i} - b_{\text{ox}})^2 + \sum_{j} w_{j,\text{red}}(j_{\text{red},j} - mE_{\text{red},j} - b_{\text{red}})^2$$

where weights $w_{i,\text{ox}}$ and $w_{j,\text{red}}$ were set to unity for all points (equal weighting). The resulting fit parameters were determined via QR decomposition of the weighted design matrix.

#### Potential Windows

The analysis was performed within the following potential windows:
- **Oxidation branch (forward scan):** 1.00–1.05 V vs RHE
- **Reduction branch (backward scan):** 1.00–1.05 V vs RHE

These windows were selected to capture the catalytically relevant potential region where the material exhibits significant redox activity.

#### Offset Parameter

The vertical offset between the two fitted lines—the primary metric reported—was defined as:
$$\Delta b = b_{\text{ox}} - b_{\text{red}}$$

This parameter directly quantifies the hysteresis: positive values indicate that the oxidation branch carries higher current at equivalent potentials compared to the reduction branch.

### Outlier Detection and Removal

A statistical approach was employed to identify anomalous offset values that deviate significantly from the trend. Offset values at regularly sampled cycles (every 10th cycle, starting from cycle 8) were subjected to z-score analysis:
$$z = \frac{\Delta b - \bar{\Delta b}}{\sigma}$$

where $\bar{\Delta b}$ is the mean offset and $\sigma$ is the standard deviation of the sampled population. Data points with $|z| > 50$ were classified as outliers and excluded from visualization and subsequent analysis. This threshold was chosen to preserve genuine features while removing spurious measurement artifacts.

### Data Aggregation

For multi-sample studies, the offset data from all processed directories were consolidated into wide-format CSV files, with rows corresponding to cycle numbers and columns containing the offset, slope, and intercept parameters for each sample. Two complementary files were generated:
1. **Complete dataset:** All fitted cycles from all samples
2. **Sampled dataset:** Only the regularly sampled cycles (every 10th) from all samples

The second file enables efficient comparison of samples without the confounding effects of high temporal resolution variations.

## Figure Generation

Representative cyclic voltammograms with overlaid double-line fits were generated for up to 10 equally-spaced cycles from each sample. These plots display:
- The complete forward and backward scan current profiles
- The potential windows used for fitting (highlighted markers)
- Extrapolated linear fits extending 50% beyond the window boundaries
- Annotated offset values ($\Delta b$) for quick reference

Contour plots of current density vs. potential vs. cycle number were generated at high resolution to visualize long-term electrochemical stability and potential drift over 1000 cycles.

## Software and Implementation

Data processing was performed using Python 3.10 with the following libraries:
- **pandas** (data manipulation and I/O)
- **NumPy** (numerical computations and linear algebra)
- **Matplotlib** (figure generation)
- **SciPy** (interpolation routines)
- **CMCrameri** (perceptually uniform colormaps)

The complete analysis pipeline was implemented as a modular script capable of batch-processing multiple electrochemical datasets from different catalyst compositions and preparation conditions.
