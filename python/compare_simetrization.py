import numpy as np

def simetrize_3dhistogram(histogram):
    """
        Returns the simetrized version of a 3d histogram
        Params:
            -histogram: numpy array. The histogram to be simetrized. It is assumed to be of the from NxNxN

        Returns:
            -simetrized: histogram. The same histogram but simetrized, where any combination of its indices has the very same value.
    """
    N =len(histogram)
    n_histogram = np.zeros((N,N,N))
    for i in range(N-2):
        for j in range(i+1,N-1):
            for k in range(j+1, N):
                S = histogram[i][j][k] + histogram[k][i][j] + histogram[j][k][i] + histogram[i][k][j] + histogram[j][i][k] + histogram[k][j][i]
                n_histogram[i][j][k] = S
                n_histogram[k][i][j] = S
                n_histogram[j][k][i] = S
                n_histogram[i][k][j] = S
                n_histogram[j][i][k] = S
                n_histogram[k][j][i] = S
                #a[i][j][k], a[k][i][j], a[j][k][i], a[i][k][j], a[j][i][k], a[k][j][i]
    return n_histogram

def simetrizeinteral_3dhistogram(fake_sample,d_max,bins_number):
    XYZ = np.zeros((30,30,30))
    N = len(fake_sample)
    n=0
    dbin = d_max/bins_number
    for triangle in fake_sample:
        if triangle[0]<=d_max and triangle[1]<=d_max and triangle[2]<=d_max:

            n+=1
            a,b,c = (int(triangle[0]/dbin),int(triangle[1]/dbin),int(triangle[2]/dbin))
            XYZ[a,b,c]+=1
            XYZ[a,c,b]+=1
            XYZ[b,a,c]+=1
            XYZ[b,c,a]+=1
            XYZ[c,b,a]+=1
            XYZ[c,a,b]+=1
    print(n)
    return XYZ


fake_sample = np.random.rand(1000000,3)*400

d_max = 180
bins_number = 30


XYZ, edges = np.histogramdd(fake_sample, bins=(bins_number,bins_number,bins_number), range=[[0,d_max],[0,d_max],[0,d_max]])
sim2_XYZ = simetrize_3dhistogram(XYZ) #simetrización externa

sim1_XYZ = simetrizeinteral_3dhistogram(fake_sample,d_max,bins_number) #simetrización interna


print(sim1_XYZ)
print(sim2_XYZ)

sim1_fails = []
sim2_fails = []

for i in range(10000):
    a,b,c = np.random.randint(0,30),np.random.randint(0,30),np.random.randint(0,30)
    # for b in range(30):
    #     for c in range(30):
    if not sim1_XYZ[a,b,c]==sim1_XYZ[a,c,b]==sim1_XYZ[b,a,c]==sim1_XYZ[b,c,a]==sim1_XYZ[c,b,a]==sim1_XYZ[c,a,b]:
        print("Usando la simetrización interna")
        print(f'No son simetricos en {(a,b,c)}')
        sim1_fails += [(a,b,c)]
    if not sim2_XYZ[a,b,c]==sim2_XYZ[a,c,b]==sim2_XYZ[b,a,c]==sim2_XYZ[b,c,a]==sim2_XYZ[c,b,a]==sim2_XYZ[c,a,b]:
        print("Usando la simetrización externa")
        print(f'No son simetricos en {(a,b,c)}')
        sim2_fails += [(a,b,c)]

print(f'Los histogramas son iguales? {(sim1_XYZ==sim2_XYZ).all()}')