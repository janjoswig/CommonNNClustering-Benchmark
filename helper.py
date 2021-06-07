import json

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from cnnclustering import cluster


class RecordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, cluster.Record):
            return {
                "_type": "Record",
                "n_points": obj.n_points,
                "radius_cutoff": obj.radius_cutoff,
                "cnn_cutoff": obj.cnn_cutoff,
                "member_cutoff": obj.member_cutoff,
                "max_clusters": obj.max_clusters,
                "n_clusters": obj.n_clusters,
                "ratio_largest": obj.ratio_largest,
                "ratio_noise": obj.ratio_noise,
                "execution_time": obj.execution_time,
                }
        return super().default(self, obj)


def as_Record(obj):
    try:
        _type = obj.pop('_type')
    except KeyError:
        _type = None

    if _type is None:
        return obj

    if _type == 'Record':
        return cluster.Record(*obj.values())

    return obj


def compute_neighbours(data, radius, sort=False):

    tree = KDTree(data)
    neighbourhoods = tree.query_radius(
        data, r=radius, return_distance=False
        )

    if sort:
        for n in neighbourhoods:
            n.sort()

    return neighbourhoods


def gen_no_structure_points(size, random_state=2021):
    rng = np.random.default_rng(random_state)
    no_structure = rng.random(size)

    return StandardScaler().fit_transform(no_structure)


def gen_blobs_points(size, random_state=8, **kwargs):
    blobs, _ = datasets.make_blobs(
        n_samples=size[0],
        n_features=size[1],
        random_state=random_state,
        **kwargs
        )
    return StandardScaler().fit_transform(blobs)


def gen_circles_points(size, random_state=8, **kwargs):
    if size[1] != 2:
        raise RuntimeError("Can only generate circles in 2D.")

    circles, _ = datasets.make_circles(
        n_samples=size[0],
        random_state=random_state,
        **kwargs
        )
    return StandardScaler().fit_transform(circles)


def gen_moons_points(size, random_state=8, **kwargs):
    if size[1] != 2:
        raise RuntimeError("Can only generate circles in 2D.")

    circles, _ = datasets.make_moons(
        n_samples=size[0],
        random_state=random_state,
        **kwargs
        )
    return StandardScaler().fit_transform(circles)