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
            point_array = array('d', point)
            point_array.tofile(file)

    # write labels
    with open(labels_fp, 'wb') as file:
        label_array = array('l', labels)
        label_array.tofile(file)

    return data


def read_data(filepath, num_data, num_features, data_format='d'):
    vals = [[] for _ in range(num_data)]
    with open(filepath, 'rb') as file:
        data = file.read(8)
        i = 0
        while data:
            ind = i // num_features
            # if ind == 10:
            #     break
            vals[ind].append(struct.unpack(data_format, data)[0])
            data = file.read(8)
            i += 1
    ret = np.concatenate([np.array(i) for i in vals]).reshape((-1, num_features))
    return ret[~np.isnan(ret)].reshape((-1, 2))


def read_clustering(filepath):
    vals = []
    with open(filepath, 'rb') as file:
        data = file.read(4)
        while data:
            vals.append(struct.unpack('i', data)[0])
            data = file.read(4)

    return vals


def plot_data(data, clusters, clustering):
    colors = cm.jet(np.linspace(0, 1, len(clusters)))
    new_colors = [colors[i] for i in clustering]
    plt.scatter(data[:, 0], data[:, 1], c=new_colors)
    plt.scatter(clusters[:, 0], clusters[:, 1], c='black')

    plt.show()


NUM_DATA = 1000000
NUM_FEATURES = 10
NUM_CLUSTERS = 50
CLUSTER_STD = 1
BOX = (-100, 100)

# data = generate_data(NUM_DATA, NUM_FEATURES, NUM_CLUSTERS, CLUSTER_STD, BOX,
#                      f'data_{NUM_DATA}_{NUM_FEATURES}.txt', f'data_labels_{NUM_DATA}_{NUM_FEATURES}.txt')


num = 200000
dims = 2
num_clusters = 30
file_num = 0
data = read_data(f'data_{num}_{dims}_{num_clusters}.txt', num, dims)
clusters = read_data(f'data_{num}_{dims}_{num_clusters}_clusters_{file_num}.txt', num, dims)
clustering = read_clustering(f'data_{num}_{dims}_{num_clusters}_clustering_{file_num}.txt')
plot_data(data, clusters, clustering)
