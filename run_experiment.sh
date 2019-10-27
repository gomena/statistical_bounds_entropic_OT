#!/bin/sh

for n_chain in {1..800}
do
for noise_level in 0.1 1
do
for d in 1 5 10
do
python experiment_entropy.py -d $d -n_chain $n_chain -noise_level $noise_level
done
done
done
