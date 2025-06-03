# Quantile Factor Models: MATLAB Implementation

This repository contains MATLAB code to replicate the empirical results from two related papers on quantile factor models by Dimitris Korobilis and Maximilian Schröder.

## Papers

### 1. Probabilistic Quantile Factor Analysis
**Citation**: Korobilis, D. and Schröder, M. (forthcoming). "Probabilistic Quantile Factor Analysis," *Journal of Business and Economic Statistics*.

**Code Location**: [`VBQFA/`](VBQFA/)

This paper introduces a novel Bayesian approach to quantile factor analysis using variational inference methods.

### 2. Monitoring Multi-Country Macroeconomic Risk: A QFAVAR Approach  
**Citation**: Korobilis, D. and Schröder, M. (2024). "Monitoring multi-country macroeconomic risk: A quantile factor-augmented vector autoregressive (QFAVAR) approach," *Journal of Econometrics*, 105730. https://doi.org/10.1016/j.jeconom.2024.105730

**Code Location**: [`QFAVAR/`](QFAVAR/)

This paper extends quantile factor analysis to vector autoregressions for monitoring macroeconomic risk across multiple countries.

## Repository Structure

```
quantile-factor-models/
├── README.md                                    # This file
├── LICENSE
├── VBQFA/                                      # JBES paper
│   ├── README.md
│   ├── Empirical_1/                           # Uncertainty data analysis
│   ├── Empirical_2/                           # FCI data analysis
│   ├── MonteCarlo_ELBO/                       # Factor selection via ELBO
│   ├── MonteCarlo_MAIN/                       # Algorithm comparison vs IQR
│   └── shared-functions/                       # Common utilities
├── QFAVAR/                                     # Journal of Econometrics paper
│   ├── README.md
│   ├── Forecasting/                           # Forecasting exercises
│   │   ├── FORECASTING_0_QFAVAR.m            # QFAVAR vs QFAVAR-SV
│   │   ├── FORECASTING_1_vsFAVAR.m           # QFAVAR vs FAVAR
│   │   ├── FORECASTING_0_vsStochvol.m        # Multiple model comparison
│   │   └── FORECASTING_0_QFAVAR.m            # QFAVAR vs univariate QR
│   ├── Structural/                            # Structural analysis
│   │   ├── FAVAR_GIRFs.m                     # FAVAR impulse responses
│   │   └── QFAVAR_GIRFs.m                    # QFAVAR impulse responses
│   └── shared-functions/                       # Common utilities
└── docs/                                       # Additional documentation
```

## Quick Start

### Requirements
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox
- Econometrics Toolbox (recommended)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/[username]/quantile-factor-models.git
   cd quantile-factor-models
   ```

2. Add the repository to your MATLAB path:
   ```matlab
   addpath(genpath('.'))
   ```

### Running the Code

#### Probabilistic Quantile Factor Analysis (JBES)
```matlab
% Navigate to the paper folder
cd VBQFA

% Run empirical exercises
cd Empirical_1
% [Run main script - see folder README for details]

cd ../Empirical_2  
% [Run main script - see folder README for details]

% Run Monte Carlo exercises
cd ../MonteCarlo_MAIN
% [Run main script - see folder README for details]
```

#### QFAVAR Macroeconomic Risk Monitoring (Journal of Econometrics)
```matlab
% Navigate to the paper folder
cd QFAVAR

% Run forecasting comparisons
cd Forecasting
run FORECASTING_0_QFAVAR    % Compare QFAVAR models
run FORECASTING_1_vsFAVAR   % Compare with FAVAR
run FORECASTING_0_vsStochvol % Compare with SV models

% Run structural analysis
cd ../Structural
run FAVAR_GIRFs             % FAVAR impulse responses  
run QFAVAR_GIRFs            % QFAVAR impulse responses
```

## Method Overview

**Quantile Factor Analysis** extracts latent factors that explain the conditional quantiles of observed variables, providing a more complete picture of distributional relationships than traditional factor models that focus only on conditional means.

**QFAVAR** extends this framework to vector autoregressions, enabling:
- Multi-country macroeconomic risk monitoring
- Quantile-specific impulse response analysis
- Improved forecasting across the conditional distribution

## Key Features

- **Bayesian Inference**: Variational Bayes algorithms for efficient estimation
- **Factor Selection**: Automatic relevance determination via ELBO optimization
- **Flexible Error Distributions**: Accommodates various distributional assumptions
- **Forecasting**: Multiple model comparisons and evaluation metrics
- **Structural Analysis**: Generalized impulse response functions
- **Monte Carlo Validation**: Extensive simulation studies

## Data Requirements

The code is designed to work with:
- **Uncertainty indicators** (Empirical exercise 1)
- **Financial Conditions Index (FCI)** data (Empirical exercise 2)  
- **Multi-country macroeconomic variables** (QFAVAR applications)

*Note: Due to data licensing restrictions, users need to obtain data from original sources. See individual folder READMEs for data source information.*

## Computational Notes

- Monte Carlo exercises may take several hours depending on system specifications
- Large-scale QFAVAR estimation is computationally intensive
- Consider using MATLAB Parallel Computing Toolbox for faster execution

## Citation

If you use this code in your research, please cite the relevant paper(s):

**For Quantile Factor Analysis:**
```
Korobilis, D. and Schröder, M. (forthcoming). "Probabilistic Quantile Factor Analysis," 
Journal of Business and Economic Statistics.
```

**For QFAVAR:**
```
Korobilis, D. and Schröder, M. (2024). "Monitoring multi-country macroeconomic risk: 
A quantile factor-augmented vector autoregressive (QFAVAR) approach," 
Journal of Econometrics, 105730. https://doi.org/10.1016/j.jeconom.2024.105730
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Dimitris Korobilis** - University of Glasgow
- **Maximilian Schröder** - European Central Bank

## Contact

For questions about the code or methods, please contact:
- Dimitris Korobilis: dimitris.korobilis@glasgow.ac.uk

## Issues and Contributions

Please report any issues or bugs through the GitHub issue tracker. Contributions and improvements are welcome via pull requests.

---

*This repository provides research code for academic purposes. While we strive for accuracy, please verify results for your specific applications.*
