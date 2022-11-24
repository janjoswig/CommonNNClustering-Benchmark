import sys

from commonnn import cluster
from commonnnbm import base
from sklearn.metrics import pairwise_distances


@profile
def profiled(data):
    clustering = cluster.Clustering(
        data, recipe="distances", preparation_hook="pass"
        )
    clustering.fit(radius_cutoff=0.25, similarity_cutoff=0)


if __name__ == "__main__":
    data = base.gen_no_structure_points(size=(int(sys.argv[1]), 2))
    data = pairwise_distances(data)
    profiled(data)
