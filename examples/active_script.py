#!/usr/bin/env python
''' Any argument will cause simulation, otherwise just plots '''
import sys
import numpy as np

ns = np.logspace(3, 5.8, 30, base=10.0).astype(int)[::-1]
# ns = [16, 128, 1024]
sims = {
    'dl': 'Nengo-DL',
    'ocl': 'Nengo-OCL',
    # 'crit': 'Pycriticu',
    'ref': 'Reference',
}

filenames = {st: 'record_{}.yml'.format(sn) for st, sn in sims.items()}
ns_str = ','.join(str(n) for n in ns)

if len(sys.argv) > 1 and sys.argv[1] == 'simulate':
    for simtoken in sims.keys():
        sys.argv = [
            'benchmark_wattsstrogatz.py',
            simtoken,
            ns_str,
            sims[simtoken],
            filenames[simtoken]
        ]
        with open('benchmark_wattsstrogatz.py') as fx:
            exec(fx.read())

sys.argv = ['view_records.py'] + list(filenames.values())
with open('view_records.py') as fx:
    exec(fx.read())
