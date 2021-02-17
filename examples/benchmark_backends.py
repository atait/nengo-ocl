#!/usr/bin/env python
""" Performs a benchmark simulation across multiple backends.

    To cause simulation, give "simulation" as argument. Otherwise this just plots stored data.

    nns: number of neurons for each simulation in the benchmark
    sims: list of tuple(argument-passed-to-benchmark-script, name-for-the-benchmark)

    Stores results in files with particular names. If one fails, you can comment out lines
    corresponding to simulations already completed. Perhaps change the number of neurons.
    At the end, uncomment all the lines and call without an argument to plot all the data.
"""
import os
import sys
import numpy as np

##### User modifies this section often
benchmark_script = "benchmark_wattsstrogatz.py"
nns = np.logspace(7, 20, 10, base=2).astype(int)[::-1]
# nns = [128, 1024]
# nns = 2 ** np.arange(10, 19)[::-1]
sims = [
    ("dl", "Nengo-DL"),
    ("ocl", "Nengo-OCL-csr"),
    ("ocl", "Nengo-OCL-ell"),
    ("ref", "Reference"),
]
# Use to compare ELLPACK implementations
from nengo_ocl.clra_gemv import algostr_to_planner
# sims += [("ocl", algo) for algo in algostr_to_planner.keys()]

name2file = lambda name: "og_data_100/record_{}.yml".format(
    name
)  # you can use slashes to load/save from subdirectories
#####

if len(sys.argv) > 1 and sys.argv[1] == "simulate":
    for simtoken, simname in sims:
        sys.argv = [
            benchmark_script,
            simtoken,
            ",".join(str(n) for n in nns),
            simname,
            name2file(simname),
        ]
        ### set backend-specific environment variables here
        os.environ["NENGO_OCL_SPMV_ALGORITHM"] = simname
        # "ELLPACK" if simname.endswith("ell") else "CSR"
        ###
        with open(benchmark_script) as fx:
            exec(fx.read())

sys.argv = ["view_records.py"] + [name2file(simname) for _, simname in sims]
with open("view_records.py") as fx:
    exec(fx.read())
