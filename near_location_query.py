from sklearn.neighbors import BallTree
from LBSNData import *
from utils import *
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


class Loc_Query_System:
    def __init__(self):
        self.coordinates = []
        self.tree = None
        self.n_nearest = None
        self.n_nearest_locs = None

    def build_tree(self, dataset):
        self.coordinates = np.zeros((len(dataset.idx2gps)-1, 2), dtype=np.float64)
        for idx, (lat, lng) in dataset.idx2gps.items():
            if idx != 0:
                self.coordinates[idx - 1] = [lat, lng]
        self.tree = BallTree(
            self.coordinates,
            leaf_size=1,
            metric='haversine'
        )

    def prefetch_n_nearest_locs(self, n_nearest):
        self.n_nearest = n_nearest
        self.n_nearest_locs = np.zeros((self.coordinates.shape[0], n_nearest), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            _, n_nearest_locs = self.tree.query(trg_gps, n_nearest + 1)
            n_nearest_locs = n_nearest_locs[0, 1:]
            n_nearest_locs += 1
            self.n_nearest_locs[idx] = n_nearest_locs

    def get_k_nearest_locs(self, trg_loc, k):
        if k <= self.n_nearest:
            k_nearest_locs = self.n_nearest_locs[trg_loc - 1][:k]
            return k_nearest_locs
        else:
            trg_gps = self.coordinates[trg_loc].reshape(1, -1)
            _, k_nearest_locs = self.tree.query(trg_gps, k + 1)
            k_nearest_locs = k_nearest_locs[0, 1:]
            k_nearest_locs += 1
            return k_nearest_locs

    def save(self, path):
        data = {
            "coordinates": self.coordinates,
            "tree": self.tree,
            "n_nearest": self.n_nearest,
            "n_nearest_locs": self.n_nearest_locs
        }
        serialize(data, path)

    def load(self, path):
        data = unserialize(path)
        self.coordinates = data["coordinates"]
        self.tree = data["tree"]
        self.n_nearest = data["n_nearest"]
        self.n_nearest_locs = data["n_nearest_locs"]