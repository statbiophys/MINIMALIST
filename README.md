## MINIMALIST

Written by Giulio Isacchini, MPIDS Göttingen - ENS Paris and Natanael Spisak, ENS Paris

The code is written in Python3.

Last updated on 24-05-2021

Reference: MINIMALIST: Mutual INformatIon Maximisation for Amortized Likelihood Inference from Sampled Trajectories, Giulio Isacchini, Natanel Spisak, Armita Nourmohammad, Aleksandra M. Walczak, Thierry Mora

### To reproduce the figures

In order to reproduce the plots you need to run the following commands.

1) Install the mimsbi package

Enter the mimsbi folder and run the command `python setup.py install`

2) Run analysis

To run the analysis for the the 4 task run the script run_analysis.py with the options: `ou`,`bd`,`sir` or `lorenz`. 

3) Plot the results.

For Figure 2, run `fig2.py`

For Figure 3 run `fig3.py`

For Figure 4, run `fig4.py`

#### Requisites

- `tensorflow>2.1`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `tqdm`

### To do more

This directory includes a stable version of the `mimsbi` package.  <!-- The full package is available in ... -->
