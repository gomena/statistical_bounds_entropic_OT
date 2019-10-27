# Code for "Statistical bounds for entropic optimal transport"

## Gonzalo Mena and Jonathan Niles-Weed.

This directory contains all data used to generate Figure 1 and Figure 2 of our paper.

* lib.py is a library containing all functions used to compute relevant estimators (e.g. Sinkhorn loss, mixture of gaussian loss, etc)
* experiment_entropy.py is a script that computes the three estimators ($h_{m.g}, h_{paired}, h_{ind}$) for many samples, given a particular parameter configuration. This estimator is computed at several values of $n$m between $n=100$ and $n=15000$ and for $m=n*(1-\lambda)/(\lambda)$ with $\lambda=0.3,0.5,0.7$ (for the paper only $\lambda=0.5$ is relevant. Results of the experiment are stored on a .npy file on the results folder
* run_experiment.sh is a script that runs experiment_entropy.py for all parameter configurations that are shown in Figures 1 and Figure 2.
* Results.zip contains all the results from the above script. This results were computed on a cluster using several cores in parallel, as the experiments can be very expensive, mostly for large $m,n$
* PlotResults.ipynb is a notebook that loads results from the Results directory (unzipped Result.zip) folder and creates the paper figures.
