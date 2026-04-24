# Learning To Validate Generative Models: a Goodness-of-Fit Approach

This repository provides the code required to reproduce the tables and plots from the paper:  
**[Learning to Validate Generative Models: a Goodness-of-Fit Approach](https://arxiv.org/abs/2511.09118)**


---

## Reproducing the Results

### Compute p-values and Z-scores
Use the notebook:
- [compute_Zscores.ipynb](compute_Zscores.ipynb)

### Plot Z-scores vs. sample size
Run the following scripts:
- [plotZscore_vs_N_JetData.py](plotZscore_vs_N_JetData.py)
- [plotZscore_vs_N_MoG.py](plotZscore_vs_N_MoG.py)

### Validate the null distribution
To assess the compatibility of the test statistic with a $\chi^2$ distribution, use:
- [KS_test_chi2.ipynb](KS_test_chi2.ipynb)

---


## Running the NPLM Test on 4D MoG Data

To reproduce the Mixture-of-Gaussians (MoG) experiments:

1. Clone the NPLM implementation repository:  
   *(add link)*

2. Copy the dataset:
   - From this repository: `data/4d_MoG`
   - Into: `datasets/` in the NPLM repository

3. Copy the experiment script:
   - From: `scripts/MoG_experiments.py`
   - Into: `examples/` in the NPLM repository

4. Run the experiment from the root of the NPLM repository:
   ```bash
   python -m examples.MoG_experiments


## Additional Resources

- NPLM test implementation: *(add link)*
  
- Datasets used in the study: https://zenodo.org/records/19631150


## Contributors

- P. Cappelli  
- G. Grosso  
- M. Letizia  
- H. Reyes-González  
