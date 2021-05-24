## MINIMALIST

Written by Giulio Isacchini, MPIDS Göttingen - ENS Paris and Natanael Spisak, ENS Paris

The code is written in Python3.

Last updated on 24-05-2021

Reference: MINIMALIST: Mutual INformatIon Maximisation for Amortised Likelihood Inference from Sampled Trajectories, Giulio Isacchini, Natanel Spisak, Armita Nourmohammad, Aleksandra M. Walczak, Thierry Mora

=== Reproduce Plots ===

In order to reproduce the plots you need to run the following commands.

1) Run analysis

To run the analysis for the the 4 task run the script run_analysis.py with the options: ou,bd,sir and lorentz. 

2) Plot the results.

To recreate Fig 2, run fig2.py

To recreate Fig 3 run fig3.py

To recreate Fig 4, run fig4.py

=== Requisites ===

- tensorflow>2.1
- numpy
- pandas
- scipy
- matplotlib
- tqdm

This directory includes a stable version of the mimsbi package. The full package is available in ...