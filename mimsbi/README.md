## MIMSBI

Written by Giulio Isacchini, MPIDS Göttingen - ENS Paris and Natanael Spisak, ENS Paris

The code is written in Python3.

Last updated on 24-05-2021

Reference: MINIMALIST: Mutual INformatIon Maximisation for Amortised Likelihood Inference from Sampled Trajectories, Giulio Isacchini, Natanel Spisak, Armita Nourmohammad, Aleksandra M. Walczak, Thierry Mora

=== Package Information ===
MIMSBI (Mutual Information Estimation for Simulation Based Inference) is a Python3 package to infer neural density estimators with tensorflow code.
The Neural density estimators can be used to evaluate DKL, MI and DJS using only samples from two distributions. Alternatively the density itself can be evaluated. In the case of Simulation Based the Inference the density takes the role of the posterior of the parameters given the observed data. See ref for more details.


=== Requisites ===

- tensorflow>2.1
- numpy
- pandas
- scipy
- matplotlib
- tqdm

This directory includes a stable version of the mimsbi package. The full package is available in ...