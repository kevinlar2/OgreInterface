import numpy as np


def reduce_vectors_zur_and_mcgill(vectors: np.ndarray):
    n_vectors = len(vectors)
    reduced = np.zeros(n_vectors).astype(bool)
    mats = np.repeat(np.eye(3).reshape((1, 3, 3)), n_vectors, axis=0)

    while not reduced.all():
        ui = np.where(np.logical_not(reduced))[0]
        uv = vectors[ui]
        umats = mats[ui]

        dot_str = "ij,ij->i"
        dot = np.einsum(dot_str, uv[:, 0], uv[:, 1])

        a_norm = np.einsum(dot_str, uv[:, 0], uv[:, 0])
        b_norm = np.einsum(dot_str, uv[:, 1], uv[:, 1])
        b_plus_a_norm = np.einsum(
            dot_str, uv[:, 1] + uv[:, 0], uv[:, 1] + uv[:, 0]
        )
        b_minus_a_norm = np.einsum(
            dot_str, uv[:, 1] - uv[:, 0], uv[:, 1] - uv[:, 0]
        )

        c1 = dot < 0
        c2 = a_norm > b_norm
        c3 = b_norm > b_plus_a_norm
        c4 = b_norm > b_minus_a_norm

        nc1 = np.logical_not(c1)
        nc2 = np.logical_not(c2)
        nc3 = np.logical_not(c3)
        nc4 = np.logical_not(c4)

        if c1.any():
            uv[c1, 1] *= -1
            umats[c1, 1] *= -1

        op2 = np.logical_and(nc1, c2)
        if op2.any():
            uv[op2] = uv[op2][:, [1, 0]]
            umats[op2] = umats[op2][:, [1, 0, 2]]

        op3 = np.logical_and(np.c_[nc1, nc2].all(axis=1), c3)
        if op3.any():
            uv[op3, 1] = uv[op3][:, 1] + uv[op3][:, 0]
            umats[op3, 1] = umats[op3][:, 1] + umats[op3][:, 0]

        op4 = np.logical_and(np.c_[nc1, nc2, nc3].all(axis=1), c4)
        if op4.any():
            uv[op4, 1] = uv[op4][:, 1] - uv[op4][:, 0]
            umats[op4, 1] = umats[op4][:, 1] - umats[op4][:, 0]

        reduced_inds = np.c_[nc1, nc2, nc3, nc4].all(axis=1)
        if reduced_inds.any():
            reduced[ui[reduced_inds]] = True

        vectors[ui] = uv
        mats[ui] = umats

    return vectors, mats


if __name__ == "__main__":
    from pymatgen.analysis.interfaces.zsl import reduce_vectors
    import time

    vec1 = np.array([[1, 0, 0], [-1, 1, 0]])
    vec2 = np.array([[2, 0, 0], [-1, 1, 0]])
    vec3 = np.array([[4, 0, 0], [-1, 1, 0]])
    vecs = np.stack([vec1, vec2, vec3], axis=0)
    vecs = np.repeat(vecs, 1000, axis=0)

    s = time.time()
    for i in range(100):
        vecs_out, mats_out = reduce_vectors_zur_and_mcgill(np.copy(vecs))
    print((time.time() - s) / 100)

    s = time.time()
    for i in range(100):
        for v in vecs:
            reduce_vectors(v[0], v[1])
    print((time.time() - s) / 100)
