import numpy as np
from scipy.spatial.distance import pdist, cdist

def compute_2iso(h):
    """
    Computes the 2iso pcf using cdiss and stores the results in three histograms
    """
    DATA_PATH = "/home/jaarpa/3PCFGPU/data/data_aa.dat"
    DATA_PIPS_PATH = "/home/jaarpa/3PCFGPU/data/data_aa.pip"
    RAND_PATH = "/home/jaarpa/3PCFGPU/data/rand_aa.dat"
    MAX_DISTANCE = 100
    BINS = 40

    data = np.loadtxt(DATA_PATH)
    random_data = np.loadtxt(RAND_PATH)

    if h=="rr":
        r_distances = pdist(random_data)
        rr = np.histogram(r_distances, bins=BINS, range=(0,MAX_DISTANCE))
        np.savetxt("rr.dat", rr[0], fmt='%1.4f')
    elif h=="dd":
        d_distances = pdist(data)
        dd = np.histogram(d_distances, bins=BINS, range=(0,MAX_DISTANCE))
        np.savetxt("dd.dat", dd[0], fmt='%1.4f')
    elif h=="dr":
        dr_distances = cdist(data, random_data)
        dr = np.histogram(dr_distances, bins=BINS, range=(0,MAX_DISTANCE))
        np.savetxt("dr.dat", dr[0], fmt='%1.4f')

if __name__ == "__main__":
    compute_2iso("dd")
