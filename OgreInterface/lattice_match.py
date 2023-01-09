import numpy as np


class ZurMcGill:
    def __init__(
        self,
        film_vectors: np.ndarray,
        substrate_vectors: np.ndarray,
        max_area: float = 400.0,
        max_linear_strain: float = 0.01,
        max_angle_strain: float = 0.01,
        max_area_mismatch: float = 0.09,
    ):
        self.film_vectors = film_vectors
        self.substrate_vectors = substrate_vectors
        self.max_area = max_area
        self.max_linear_strain = max_linear_strain
        self.max_angle_strain = max_angle_strain
        self.max_area_mismatch = max_area_mismatch

        self.film_area = self._get_area(film_vectors)
        self.substrate_area = self._get_area(substrate_vectors)
        self.area_ratio = self.film_area / self.substrate_area
        self.film_rs, self.substrate_rs = self._get_rs()

    def _get_area(self, vectors: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.cross(vectors[0], vectors[1]))

    def _get_rs(self):
        film_rs = np.arange(1, self.max_area // self.film_area)
        substrate_rs = np.arange(1, self.max_area // self.substrate_area)

        return film_rs, substrate_rs

    def _get_transformation_matrices(self, n):
        mats = [
            (i, j)
            for i in self.film_rs
            for j in self.substrate_rs
            if np.absolute(self.film_area / self.substrate_area - float(j) / i)
            < self.max_area_mismatch
        ]
        for m in mats:
            print(m)
        # factors = self._get_factors(n)
        # print(factors)

    def _get_factors(self, n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)

        return factors


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
    vectors1 = np.array([[5, 0, 0], [0, 5, 0]])
    vectors2 = np.array([[7.5, 0, 0], [0, 7.5, 0]])
    zm = ZurMcGill(film_vectors=vectors1, substrate_vectors=vectors2)
    zm._get_transformation_matrices(12)
