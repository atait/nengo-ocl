#!/usr/bin/bash
# ns_neurons=1000,3000,10_000,30_000,100_000,300_000,1_000_000,3_000_000
# python benchmark_wattsstrogatz.py dl ${ns_neurons} TensorFlow records_wattsstrogatz.yml
# python benchmark_wattsstrogatz.py ocl ${ns_neurons} Nengo-OCL tmpfile.yml
# cat tmpfile.yml >> records_wattsstrogatz.yml
# python benchmark_wattsstrogatz.py crit ${ns_neurons} Numba-CUDA tmpfile.yml
# cat tmpfile.yml >> records_wattsstrogatz.yml
# rm tmpfile.yml

ns_neurons=3000000
python benchmark_wattsstrogatz.py ref ${ns_neurons} CPU tmpfile.yml
# cat tmpfile.yml >> records_wattsstrogatz.yml
# rm tmpfile.yml
