## MINIMALIST

Written by Giulio Isacchini, MPIDS Göttingen - ENS Paris and Natanael Spisak, ENS Paris

The code is written in Python3. Last updated on 24-05-2021

Reference: MINIMALIST: Mutual INformatIon Maximisation for Amortized Likelihood Inference from Sampled Trajectories, Giulio Isacchini, Natanel Spisak, Armita Nourmohammad, Thierry Mora and Aleksandra M. Walczak

### To reproduce the figures

In order to reproduce the plots you need to run the following commands.

1) Install the `mimsbi` package

Enter the mimsbi folder and run the command `python setup.py install`

2) Run analysis

To run the analysis for the the 4 task run the script `run_analysis.py` with the options: `ou`,`bd`,`sir` and/or `lorenz`. 

3) Plot the results.

For Figure 2, run `fig2.py`

For Figure 3 run `fig3.py`

For Figure 4, run `fig4.py`


### To do more

This directory includes a stable version of the `mimsbi` package.  <!-- The full package is available in ... -->

The package allows to infer the likelihood-to-evidence ratio model using one of three objective functions: MINE, FDIV or BCE. The package has implemented simulators for the processes studied in the MINIMALIST paper: Ornstein-Uhlenbeck, birth-death, SIR and Lorenz processes. To add another functionality one needs to add a new `Simulator` class to `mimsbi/models`. Then, inference can be performed using the `DensityRatioEstimator` class. For example of usage go to the `scripts` directory where separate files can be used to 
1) simulate the data  `scripts/simulate.py`
2) tune network hyperparameters  `scripts/infer_hyperpars.py`
3) likelihood-to-evidence ratio inference  `scripts/infer_estimators.py`
4) posterior evaluation  `scripts/compare_estimators.py`

To use the scripts with a new model, its specifications need to be added in `scripts/utils.py` `return_pars` function. A simple data generation to posterior evaluation protocol is also available in the `mimsbi/tutorial.ipynb` Jupyter notebook.

### Requisites

- `tensorflow>2.1`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `tqdm`
