from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from array import array
import struct
import numpy as np


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


def read_data(filepath, num_data, num_features):
    vals = [[] for _ in range(num_data)]
    with open(filepath, 'rb') as file:
        data = file.read(4)
        i = 0
        while data:
            ind = i // num_features
            # if ind == 10:
            #     break
            vals[ind].append(struct.unpack('f', data)[0])
            data = file.read(4)
            i += 1
    ret = np.concatenate([np.array(i) for i in vals]).reshape((-1, num_features))
    return ret[~np.isnan(ret)].reshape((-1, 2))


def plot_data(data, clusters, clustering):
    print(clusters)
    plt.scatter(data[:, 0], data[:, 1], c='b')
    plt.scatter(clusters[:, 0], clusters[:, 1], c='r')
    plt.show()
    plt.savefig('test2.png')


NUM_DATA = 10000
NUM_FEATURES = 2
NUM_CLUSTERS = 30
CLUSTER_STD = 10
BOX = (-1000, 1000)
# generate_data(NUM_DATA, NUM_FEATURES, NUM_CLUSTERS, CLUSTER_STD, BOX,
#               f'test_{NUM_DATA}_{NUM_FEATURES}.txt', f'test_labels_{NUM_DATA}_{NUM_FEATURES}.txt')

data = read_data('test_10000_2.txt', 10000, 2)
# clusters = read_data('clusters_scale.txt', 10000, 2)
# clustering = read_data('clustering_scale.txt', 10000, 1)
clusters = read_data('clusters_coreset_mpi.txt', 10000, 2)
clustering = read_data('clustering_coreset_mpi.txt', 10000, 1)
plot_data(data, clusters, clustering)
