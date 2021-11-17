import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm
from data_process import *


class QuerySystem:
    def __init__(self):
        self.coordinate = []
        self.tree = None
        self.knn = None
        self.knn_result = None

    def build_tree(self, dataset):
        self.coordinate = np.zeros((len(dataset.idx2gps) - 1, 2), dtype=np.float64)
        for idx, (lat, lon) in dataset.idx2gps.items():
            if idx != 0:
                self.coordinate[idx - 1] = [lat, lon]

        self.tree = BallTree(
            self.coordinate,
            leaf_size=1,
            metric='haversine'
        )

    def prefetch_knn(self, num_neighbours):
        self.knn = num_neighbours
        self.knn_result = np.zeros((self.coordinate.shape[0], num_neighbours), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.coordinate), total=len(self.coordinate), leave=True):
            tgt_gps = gps.reshape(1, -1)
            _, knn_pois = self.tree.query(tgt_gps, num_neighbours + 1)
            knn_pois = knn_pois[0, 1:]
            knn_pois += 1
            self.knn_result[idx] = knn_pois

    def get_knn(self, tgt_poi, num_nearest):
        if num_nearest <= self.knn:
            return self.knn_result[tgt_poi - 1][:num_nearest]
        tgt_gps = self.coordinate[tgt_poi - 1].reshape(1, -1)
        _, knn_pois = self.tree.query(tgt_gps, num_nearest + 1)
        knn_pois = knn_pois[0, 1:]
        knn_pois += 1
        return knn_pois

    def save(self, path):
        data = {
            'coordinates': self.coordinate,
            'tree': self.tree,
            'knn': self.knn,
            'knn_results': self.knn_result
        }
        serialize(data, path)

    def load(self, path):
        data = unserialize(path)
        self.coordinate = data['coordinates']
        self.tree = data['tree']
        self.knn = data['knn']
        self.knn_result = data['knn_results']


if __name__ == "__main__":
    filename = ' '
    path = ' '
    num_neighbours = 2000

    dataset = unserialize(filename)
    
    QuerySys = QuerySystem()
    QuerySys.build_tree(dataset)
    QuerySys.prefetch_knn(num_neighbours=num_neighbours)
    QuerySys.save(path=path)