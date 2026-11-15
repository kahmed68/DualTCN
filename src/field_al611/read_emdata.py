"""Reader for the AL611 processed TD-CSEM files (MARE2DEM EMData_2.3 format, PANGAEA doi:10.1594/PANGAEA.989421,
CC-BY 4.0, Lohrberg et al. 2026). Returns gate times, transmitters, receivers, and a data table."""
import numpy as np


def read_emdata(path):
    L = open(path).read().splitlines()
    out, i = {}, 0
    def block(tag):
        nonlocal i
        while not L[i].startswith('# ' + tag) and not L[i].startswith(tag):
            i += 1
        n = int(L[i].split(':')[1]); i += 1
        if L[i].lstrip().startswith('!'):
            i += 1
        rows = L[i:i + n]; i += n
        return rows
    hdr = {l.split(':')[0].strip(): l.split(':', 1)[1].strip() for l in L[:4] if ':' in l}
    out['header'] = hdr
    out['times'] = np.array([float(r) for r in block('CSEM Times')])
    tx = [r.split() for r in block('Transmitters')]
    out['tx'] = dict(x=np.array([float(t[0]) for t in tx]), y=np.array([float(t[1]) for t in tx]),
                     z=np.array([float(t[2]) for t in tx]), azimuth=np.array([float(t[3]) for t in tx]),
                     dip=np.array([float(t[4]) for t in tx]), length=np.array([float(t[5]) for t in tx]),
                     type=[t[6] for t in tx], name=[t[7] for t in tx])
    rx = [r.split() for r in block('CSEM Receivers')]
    out['rx'] = dict(x=np.array([float(t[0]) for t in rx]), y=np.array([float(t[1]) for t in rx]),
                     z=np.array([float(t[2]) for t in rx]), theta=np.array([float(t[3]) for t in rx]),
                     alpha=np.array([float(t[4]) for t in rx]), beta=np.array([float(t[5]) for t in rx]),
                     length=np.array([float(t[6]) for t in rx]), stepon=[t[7] for t in rx], name=[t[8] for t in rx])
    d = [r.split() for r in block('Data')]
    out['data'] = dict(type=np.array([int(t[0]) for t in d]), it=np.array([int(t[1]) for t in d]) - 1,
                       itx=np.array([int(t[2]) for t in d]) - 1, irx=np.array([int(t[3]) for t in d]) - 1,
                       value=np.array([float(t[4]) for t in d]), err=np.array([float(t[5]) for t in d]))
    return out
