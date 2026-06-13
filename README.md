# Adaptive Importance Sampling for Rare Events Estimation

This repository contains the computational implementations developed for my Master's thesis. It focuses on advanced statistical modeling and probability theory, specifically applying and comparing various Adaptive Importance Sampling methods for rare event estimation.

## Overview
The codebase includes the core engines for several advanced sampling algorithms and their application to both theoretical limit states and physical porous media simulations (including CO2 storage models).

## Repository Structure

### 1. `/algorithms`
Core implementations of the sampling methods:
* **SAIS** (Sequential Adaptive Importance Sampling)
* **iCE-IS GMM** (improved Cross-Entropy Importance Sampling with Gaussian Mixture Models)
* **SuS** (Subset Simulation)

### 2. `/applications`
Application of the core algorithms to various test cases and physical models:
* **`/artificial_limit_states`**: Testing and benchmarking on six artificial limit state functions.
* **`/1d_co2_model`**: Application to a 1D CO2 storage reliability model.
* **`/mrst_johansen_example`**: Complex application simulating the Johansen formation leveraging the MRST (MATLAB Reservoir Simulation Toolbox).