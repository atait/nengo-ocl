#!/usr/bin/env python
"""usage: python benchmark_circonv.py (ref|ocl) d1,... [name]

(ref|ocl|dl)
  Passing 'ref' will use the reference simulator, `nengo.Simulator`,
  while passing 'ocl' will use `nengo_ocl.Simulator`.

d1,...
  A comma separated list of integers referring to
  the number of dimensions in the vectors being convolved in the benchmark.
  A typical value would be 16,32,64,128,256,512.

name
  An optional name to give to the benchmark run. The name will be
  displayed in the legend plotted with `view_records.py`.
  If not given, one will be automatically generated for you.

Example usage:
  python benchmark_wattsstrogatz.py ref 2,4,6 "Reference simulator"
"""

from collections import OrderedDict
import datetime
import logging
import sys
import time
import yaml
import math

try:
    from line_profiler import LineProfiler
except ImportError:
    pass
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import numpy as np
import pyopencl as cl

import nengo
from nengo.utils.numpy import scipy_sparse

import nengo_ocl


if len(sys.argv) not in (3, 4, 5):
    print(__doc__)
    sys.exit()

if sys.argv[1] == "ref":
    sim_name = "ref" if len(sys.argv) == 3 else sys.argv[3]
    sim_class = nengo.Simulator
    sim_kwargs = dict()
elif sys.argv[1].startswith("ocl"):
    assert sys.argv[1] in ("ocl", "ocl_profile")
    profiling = sys.argv[1] == "ocl_profile"

    ctx = cl.create_some_context()
    sim_name = ctx.devices[0].name if len(sys.argv) == 3 else sys.argv[3]
    sim_class = nengo_ocl.Simulator
    sim_kwargs = dict(context=ctx, profiling=profiling, progress_bar=False, n_prealloc_probes=4)
elif sys.argv[1] == "dl":
    import nengo_dl

    sim_name = "dl" if len(sys.argv) == 3 else sys.argv[3]
    sim_class = nengo_dl.Simulator
    sim_kwargs = dict(progress_bar=True)
elif sys.argv[1] == "crit":
    sys.path.append('/home/atait/Documents/git-research/criticu')
    import pycriticu.simulator as critsim

    sim_name = "crit" if len(sys.argv) == 3 else sys.argv[3]
    sim_class = critsim
    sim_kwargs = dict(progress_bar=True)
else:
    raise Exception("unknown sim", sys.argv[1])

ns_neurons = map(int, sys.argv[2].split(","))
# ns_neurons = 10 * list(ns_neurons)  # repeat to build statistics on short simulations

### User options ###
simtime = .2
fan_outs = 16
rewire_frac = 0.2
directed = True
n_prealloc_probes = 64  # this applies to OCL and DL
sparse = True  # setting to False will probably overwhelm your GPU memory


def wattsstrogatz_adjacencies(
    n, fan_outs=fan_outs, rewire_frac=rewire_frac, directed=directed
):
    # the ring network
    offsets = list(range(1, fan_outs + 1))
    offsets.extend([offs - n for offs in offsets])
    if not directed:
        offsets.extend([-offs for offs in offsets])
    swmat = scipy_sparse.diags(
        np.ones((len(offsets), n), dtype=np.float32),
        offsets=offsets,
        shape=(n, n),
        format="csr",
    )
    # random rewires
    n_edges = len(swmat.data)
    rewires_bool = np.random.random(n_edges) < rewire_frac
    rewire_inds = rewires_bool.nonzero()[0]
    rewire_to = np.random.randint(n, size=rewire_inds.size)
    swmat.indices[rewire_inds] = rewire_to
    return swmat


records = []

for i, n_neurons in enumerate(ns_neurons):
    n_steps = math.ceil(simtime / .001)

    rng = np.random.RandomState(123)
    # a = rng.normal(scale=np.sqrt(1./dim), size=n_neurons)
    # profiler = LineProfiler()
    if sys.argv[1].startswith('ocl'):
        sim_kwargs['n_prealloc_probes'] = n_prealloc_probes

    # --- Model
    try:
        if not sys.argv[1] == "crit":
            with nengo.Network(seed=9) as model:
                if sys.argv[1] == "dl":
                    nengo_dl.configure_settings(inference_only=True)
                # inputA = nengo.Node(a)
                ens = nengo.Ensemble(n_neurons, 1)
                # nengo.Connection(inputA, ens.neurons, synapse=0.03)

                weimat = wattsstrogatz_adjacencies(n_neurons)
                if sparse:
                    transform = nengo.transforms.Sparse((n_neurons, n_neurons), init=weimat)
                else:
                    transform = weimat.toarray()
                nengo.Connection(ens.neurons, ens.neurons, transform=transform, synapse=0.1)

                # A_p = nengo.Probe(A.output, synapse=0.03)
                E_p = nengo.Probe(ens.neurons, synapse=0.03)
        else:
            weimat = wattsstrogatz_adjacencies(n_neurons)
            critsim.specify_ws(weimat, biases=np.zeros((n_steps, n_neurons), dtype=np.float32))
            critsim.setup(use_numba_kernel=True, use_cuSPARSE=False)

        # --- Simulation
        t_start = time.time()

        if not sys.argv[1] == "crit":
            # -- build
            with sim_class(model, **sim_kwargs) as sim:
                # profiler.add_function(sim._probe_outputs[E_p].step)
                # profiler.enable_by_count()
                t_sim = time.time()

                # -- warmup
                sim.run(0.01)
                t_warm = time.time()

                # -- long-term timing
                if sys.argv[1] == "dl":
                    for _ in range(n_steps // n_prealloc_probes):
                        sim.run_steps(n_prealloc_probes)
                    if n_steps % n_prealloc_probes > 0:
                        sim.run_steps(n_steps % n_prealloc_probes)
                else:
                    sim.run(simtime)
                t_run = time.time()

                if getattr(sim, "profiling", False):
                    sim.print_profiling(sort=1)
        else:
            t_sim = time.time()
            critsim.run(1)
            t_warm = time.time()
            critsim.run(n_steps)
            t_run = time.time()
        records.append(
            OrderedDict(
                (
                    ("benchmark", "watts-strogatz"),
                    ("name", sim_name),
                    ("n_neurons", n_neurons),
                    ("fan_outs", fan_outs),
                    ("simtime", simtime),
                    ("status", "ok"),
                    # ("profiling", getattr(sim, "profiling", 0)),
                    ("buildtime", t_sim - t_start),
                    ("warmtime", t_warm - t_sim),
                    ("runtime", t_run - t_warm),
                )
            )
        )
        print(records[-1])
        print("%s, n_neurons=%d successful" % (sim_name, n_neurons))
        if not sys.argv[1] == "crit":
            del model, sim
        # profiler.print_stats()
    except Exception as e:
        records.append(
            OrderedDict(
                (
                    ("benchmark", "watts-strogatz"),
                    ("name", sim_name),
                    ("n_neurons", n_neurons),
                    ("simtime", simtime),
                    ("status", "exception"),
                    ("exception", str(e)),
                    ("buildtime", t_sim - t_start),
                    ("warmtime", t_warm - t_sim),
                    ("runtime", t_run - t_warm),
                )
            )
        )
        print(records[-1])
        print("%s, n_neurons=%d exception" % (sim_name, n_neurons))
        # raise

if len(sys.argv) > 4:
    filename = sys.argv[4]
else:
    filename = "records_wattsstrogatz_%s.yml" % (
        (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
with open(filename, "w") as fh:
    yaml.dump(records, fh)
