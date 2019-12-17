from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from array import array
import struct
import numpy as np
import matplotlib.cm as cm


def generate_data(num_data, num_features, num_clusters, cluster_std, box, data_fp, labels_fp):
    data, labels = make_blobs(n_samples=num_data, n_features=num_features,
                              centers=num_clusters, cluster_std=cluster_std, center_box=box)

    #  write data
    with open(data_fp, 'wb') as file:
        for point in data:
            point_array = array('f', point)
            point_array.tofile(file)

    # write labels
    with open(labels_fp, 'wb') as file:
        label_array = array('l', labels)
        label_array.tofile(file)


def read_data(filepath, num_data, num_features, data_format='f'):
    vals = [[] for _ in range(num_data)]
    with open(filepath, 'rb') as file:
        data = file.read(4)
        i = 0
        while data:
            ind = i // num_features
            # if ind == 10:
            #     break
            vals[ind].append(struct.unpack(data_format, data)[0])
            data = file.read(4)
            i += 1
    ret = np.concatenate([np.array(i) for i in vals]).reshape((-1, num_features))
    return ret[~np.isnan(ret)].reshape((-1, num_features))


def read_clustering(filepath):
    vals = []
    with open(filepath, 'rb') as file:
        data = file.read(4)
        while data:
            vals.append(struct.unpack('i', data)[0])
            data = file.read(4)

    return vals


def plot_data(data, clusters, clustering):
    # print(data)
    # print(clustering)
    # colors = cm.jet(np.linspace(0, 1, len(set(clustering))))
    # new_colors = [colors[i] for i in clustering]
    plt.scatter(data[:, 0], data[:, 1], c='red')
    plt.scatter(clusters[:, 0], clusters[:, 1], c='black')
    plt.show()
    plt.savefig( f'test_{NUM_DATA}_{NUM_FEATURES}.png')


NUM_DATA = 1000000
NUM_FEATURES = 15
NUM_CLUSTERS = 30
CLUSTER_STD = 10
BOX = (-1000, 1000)
generate_data(NUM_DATA, NUM_FEATURES, NUM_CLUSTERS, CLUSTER_STD, BOX,
              f'test_{NUM_DATA}_{NUM_FEATURES}.txt', f'test_labels_{NUM_DATA}_{NUM_FEATURES}.txt')

<<<<<<< HEAD
data = read_data('test_10000_2.txt', 10000, 2)
<<<<<<< HEAD
clusters = read_data('clusters_serial_scale.txt', 10000, 2)
clustering = read_clustering('clustering_serial_scale.txt')
# clusters = read_data('clusters_serial_kpp.txt', 10000, 2)
# clustering = read_clustering('clustering_serial_kpp.txt')
=======
# clusters = read_data('clusters_scale.txt', 10000, 2)
# clustering = read_data('clustering_scale.txt', 10000, 1)
clusters = read_data('clusters_coreset_omp.txt', 10000, 2)
clustering = read_data('clustering_coreset_omp.txt', 10000, 1)
>>>>>>> bd0d1bb04792351d85a06e62bb7b7da6545e0ea9
plot_data(data, clusters, clustering)
=======
# data = read_data( f'test_{NUM_DATA}_{NUM_FEATURES}.txt', NUM_DATA, NUM_FEATURES)

# clusters = read_data('clusters_mpi_coresets.txt', NUM_DATA, NUM_FEATURES)
# clustering = read_clustering('clustering_mpi_coresets.txt')
# # clusters = read_data('clusters_serial_kpp.txt', 10000, 2)
# # clustering = read_clustering('clustering_serial_kpp.txt')

# plot_data(data, clusters, clustering)
>>>>>>> 214e50c... changed a few aspects of the data gen script to load the 1M point dataset and plot only two colors

# s = set()
# for x in clustering:
#     s.add(x)

# print(len(s))
