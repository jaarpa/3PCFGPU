from scipy.spatial.distance import pdist, cdist
import numpy as np
import time
import os

#Total time to enhance: 0.334310835999986

def histo(data, rand, d_max=180.0, bins_number=30, n_nodes=5):
    """
    """

    #get the box dimensions
    dnodes = np.ceil(max(max(data[:,0]),max(data[:,1]))/n_nodes)
    
    #Classificate the points
    pre_classified_data_points = [ [ [] for _ in range(n_nodes) ] for _ in range(n_nodes)]
    pre_classified_random_points = [ [ [] for _ in range(n_nodes) ] for _ in range(n_nodes)]

    for d_point, r_point in zip(data, rand):

        pre_classified_data_points[int(d_point[1]/dnodes)][int(d_point[0]/dnodes)] += [d_point]
        pre_classified_random_points[int(r_point[1]/dnodes)][int(r_point[0]/dnodes)] += [r_point]

    classified_data_points = pre_classified_data_points[0].copy()
    classified_random_points = pre_classified_random_points[0].copy()
    for i in range(1,n_nodes):
        classified_data_points += pre_classified_data_points[i].copy()
        classified_random_points += pre_classified_random_points[i].copy()

    node_labels = np.array([((i%n_nodes)*dnodes,int(i/n_nodes)*dnodes) for i in range(n_nodes**2)])

    DD_distances = []
    RR_distances = []
    DR_distances = []

    start = time.perf_counter()
    t_interDDnodal = 0
    t_interRRnodal = 0
    for i, (data_node, random_node) in enumerate(zip(classified_data_points, classified_random_points)):

        node_distance_vectors = np.abs(node_labels[i]-node_labels[i+1:]) #distance betweem every lowest left most corner of the nodes
        minimum_node_distance_vectors = node_distance_vectors-(node_distance_vectors>=1e-5)*dnodes #corrects depending on the orientation
        
        #array which has True if the distance between the node i is smaller than the maximum distance
        nodedistances_es_small = (np.sqrt(minimum_node_distance_vectors[:,0]**2+minimum_node_distance_vectors[:,1]**2))<d_max
        
        if data_node:

            DD_distances += [pdist(np.array(data_node))]

            s_start=time.perf_counter()
            for j, calculate in enumerate(nodedistances_es_small, i+1):
                if calculate and classified_data_points[j]:
                    DD_distances += [cdist(data_node,classified_data_points[j]).reshape(-1)]
            s_end=time.perf_counter()
            t_interDDnodal+=(s_end-s_start)


        if random_node:

            RR_distances += [pdist(np.array(random_node))]
            
            s_start=time.perf_counter()
            for j, calculate in enumerate(nodedistances_es_small, i+1):
                if calculate:
                    if classified_random_points[j]:
                        RR_distances += [cdist(random_node,classified_random_points[j]).reshape(-1)]

                    if classified_data_points[j]: 
                        DR_distances += [cdist(random_node,classified_data_points[j]).reshape(-1)]

            s_end=time.perf_counter()
            t_interRRnodal+=(s_end-s_start)


            if data_node:
                inner_DRnode_distances = cdist(random_node,data_node).reshape(-1) # para histograma    
                DR_distances += [cdist(random_node,data_node).reshape(-1)]

    end=time.perf_counter()
    print(f'the inter nodal distance in DD took {t_interDDnodal} s')
    print(f'the inter nodal distance in RR took {t_interRRnodal} s')
    print(f'Calculate distances in {end-start} s')

    DD_distances = np.concatenate(DD_distances)
    RR_distances = np.concatenate(RR_distances)
    DR_distances = np.concatenate(DR_distances)

    DD, b = np.histogram(DD_distances, bins=bins_number, range=(0, d_max))
    RR, b = np.histogram(RR_distances, bins=bins_number, range=(0, d_max))
    DR, b = np.histogram(DR_distances, bins=bins_number, range=(0, d_max))

    DD *= 2
    RR *= 2

    return DD, RR, DR, b




data_location='/../../fake_DATA/DATOS/data_500.dat'
rand_location='/../../fake_DATA/DATOS/rand0_500.dat'

data = np.loadtxt(fname=os.path.dirname(os.path.realpath(__file__))+data_location, delimiter=" ", usecols=(0,1))
rand = np.loadtxt(fname=os.path.dirname(os.path.realpath(__file__))+rand_location, delimiter=" ", usecols=(0,1))

start = time.perf_counter()
DD, RR, DR, edges = histo(data, rand)
end = time.perf_counter()
print(f'Total time {end-start}')

DD_BF = np.loadtxt('DD_BF.dat')
RR_BF = np.loadtxt('RR_BF.dat')
DR_BF = np.loadtxt('DR_BF.dat')

print (f'is DD correct? {DD==DD_BF}')
print (f'is RR correct? {RR==RR_BF}')
print (f'is DR correct? {DR==DR_BF}')
print (f'is DR correct? {DR<DR_BF}')

np.savetxt('DD_malla.dat', DD)
np.savetxt('RR_malla.dat', RR)
np.savetxt('DR_malla.dat', DR)