import numpy as np
from itertools import product
import time


class ZurMcGill:
    def __init__(
        self,
        film_vectors: np.ndarray,
        substrate_vectors: np.ndarray,
        max_area: float = 400.0,
        max_linear_strain: float = 0.01,
        max_angle_strain: float = 0.01,
        max_area_mismatch: float = 0.01,
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
        film_rs = np.arange(1, self.max_area // self.film_area).astype(int)
        substrate_rs = np.arange(
            1, self.max_area // self.substrate_area
        ).astype(int)

        return film_rs, substrate_rs

    def run2(self):
        equals = []
        for transforms in self._get_transformation_matrices_gen():
            film_transforms = transforms[:, 0, :, :]
            sub_transforms = transforms[:, 1, :, :]

            film_sl_vectors, sub_sl_vectors = self._get_unreduced_vectors(
                film_transforms=film_transforms, sub_transforms=sub_transforms
            )
            (
                reduced_film_sl_vectors,
                film_reduction_matrices,
                reduced_sub_sl_vectors,
                sub_reduction_matrices,
            ) = self._get_reduced_vectors(
                film_sl_vectors=film_sl_vectors, sub_sl_vectors=sub_sl_vectors
            )
            a_strain, b_strain, angle_strain, eq_inds = self._is_same(
                film_vectors=reduced_film_sl_vectors,
                sub_vectors=reduced_sub_sl_vectors,
            )

            eq_film_transforms = film_transforms[eq_inds]
            eq_sub_transforms = sub_transforms[eq_inds]
            eq_reduced_film_sl_vectors = reduced_film_sl_vectors[eq_inds]
            eq_reduced_sub_sl_vectors = reduced_sub_sl_vectors[eq_inds]
            eq_film_reduction_matrices = film_reduction_matrices[eq_inds]
            eq_sub_reduction_matrices = sub_reduction_matrices[eq_inds]
            eq_a_strain = a_strain[eq_inds]
            eq_b_strain = b_strain[eq_inds]
            eq_angle_strain = angle_strain[eq_inds]

            total_film_transformation_matrices = np.einsum(
                "...ij,...jk", eq_film_reduction_matrices, eq_film_transforms
            )
            total_sub_transformation_matrices = np.einsum(
                "...ij,...jk", eq_sub_reduction_matrices, eq_sub_transforms
            )

            equals.append(len(total_sub_transformation_matrices))

        print(np.sum(equals))

    def run(self):
        film_transforms, sub_transforms = self._get_transformation_matrices()
        film_sl_vectors, sub_sl_vectors = self._get_unreduced_vectors(
            film_transforms=film_transforms, sub_transforms=sub_transforms
        )
        (
            reduced_film_sl_vectors,
            film_reduction_matrices,
            reduced_sub_sl_vectors,
            sub_reduction_matrices,
        ) = self._get_reduced_vectors(
            film_sl_vectors=film_sl_vectors, sub_sl_vectors=sub_sl_vectors
        )

        a_strain, b_strain, angle_strain, eq_inds = self._is_same(
            film_vectors=reduced_film_sl_vectors,
            sub_vectors=reduced_sub_sl_vectors,
        )

        eq_film_transforms = film_transforms[eq_inds]
        eq_sub_transforms = sub_transforms[eq_inds]
        eq_reduced_film_sl_vectors = reduced_film_sl_vectors[eq_inds]
        eq_reduced_sub_sl_vectors = reduced_sub_sl_vectors[eq_inds]
        eq_film_reduction_matrices = film_reduction_matrices[eq_inds]
        eq_sub_reduction_matrices = sub_reduction_matrices[eq_inds]
        eq_a_strain = a_strain[eq_inds]
        eq_b_strain = b_strain[eq_inds]
        eq_angle_strain = angle_strain[eq_inds]

        total_film_transformation_matrices = np.einsum(
            "...ij,...jk", eq_film_reduction_matrices, eq_film_transforms
        )
        total_sub_transformation_matrices = np.einsum(
            "...ij,...jk", eq_sub_reduction_matrices, eq_sub_transforms
        )

        print(total_sub_transformation_matrices.shape)

    def _is_same(self, film_vectors, sub_vectors):
        film_a_norm = self._vec_norm(film_vectors[:, 0])
        film_b_norm = self._vec_norm(film_vectors[:, 1])
        sub_a_norm = self._vec_norm(sub_vectors[:, 0])
        sub_b_norm = self._vec_norm(sub_vectors[:, 1])

        a_strain = (film_a_norm / sub_a_norm) - 1
        b_strain = (film_b_norm / sub_b_norm) - 1
        a_same = np.abs(a_strain) < self.max_linear_strain
        b_same = np.abs(b_strain) < self.max_linear_strain

        angle_strain = self._angle_strain(film_vectors, sub_vectors)

        ab_angle_same = np.abs(angle_strain) < self.max_angle_strain
        strain_same = np.logical_and(a_same, b_same)

        is_equal = np.logical_and(ab_angle_same, strain_same)

        return a_strain, b_strain, angle_strain, is_equal

    def _vec_norm(self, vecs):
        dot_str = "ij,ij->i"
        norms = np.sqrt(np.einsum(dot_str, vecs, vecs))

        return norms

    def _angle_strain(self, film_vectors, sub_vectors):
        sub_angle = self._vec_angle(sub_vectors[:, 0], sub_vectors[:, 1])
        film_angle = self._vec_angle(film_vectors[:, 0], film_vectors[:, 1])

        return (sub_angle / film_angle) - 1

    def _vec_angle(self, a_vecs, b_vecs):
        dot_str = "ij,ij->i"
        cosang = np.einsum(dot_str, a_vecs, b_vecs)
        sinang = self._vec_norm(np.cross(a_vecs, b_vecs, axis=1))

        return np.arctan2(sinang, cosang)

    def _get_matching_areas(self):
        prod = product(self.film_rs, self.substrate_rs)
        prod_array = np.array(list(prod))
        film_over_sub = prod_array[:, 0] / prod_array[:, 1]
        sub_over_film = prod_array[:, 1] / prod_array[:, 0]
        film_over_sub_inds = (
            np.abs(self.area_ratio - sub_over_film) < self.max_area_mismatch
        )
        sub_over_film_inds = (
            np.abs((1 / self.area_ratio) - film_over_sub)
        ) < self.max_area_mismatch
        matching_areas = np.vstack(
            [prod_array[film_over_sub_inds], prod_array[sub_over_film_inds]]
        )
        matching_areas = np.unique(matching_areas, axis=0)
        sort_inds = np.argsort(np.prod(matching_areas, axis=1))

        matching_areas = matching_areas[sort_inds]

        return matching_areas

    def _get_transformation_matrices_gen(self):
        matching_areas = self._get_matching_areas()
        factor_dict = {
            n: self._get_factors(n) for n in np.unique(matching_areas)
        }
        for ns in matching_areas:
            yield np.array(
                list(product(factor_dict[ns[0]], factor_dict[ns[1]]))
            )

    def _get_transformation_matrices(self):
        matching_areas = self._get_matching_areas()
        factor_dict = {
            n: self._get_factors(n) for n in np.unique(matching_areas)
        }
        # TODO Need to take the product of the film and substrate transforms
        # Might be best to do it in the _get_factors functions
        for ns in matching_areas:
            yield np.array(
                list(product(factor_dict[ns[0]], factor_dict[ns[1]]))
            )
        # film_transforms = transforms[:, 0, :, :]
        # sub_transforms = transforms[:, 1, :, :]

        # print(film_transforms.shape)
        # print(sub_transforms.shape)
        # print(
        #     np.vstack(
        #         [
        #             np.array(list(product(factor_dict[4], factor_dict[9]))),
        #             np.array(list(product(factor_dict[16], factor_dict[7]))),
        #         ]
        #     ).shape
        # )
        # film_transforms = np.vstack(
        #     [factor_dict[n] for n in matching_areas[:, 0]]
        # )
        # sub_transforms = np.vstack(
        #     [factor_dict[n] for n in matching_areas[:, 1]]
        # )

        # return film_transforms, sub_transforms

    def _get_unreduced_vectors(self, film_transforms, sub_transforms):
        film_sl_vectors = np.einsum(
            "...ij,jk", film_transforms, self.film_vectors
        )
        sub_sl_vectors = np.einsum(
            "...ij,jk", sub_transforms, self.substrate_vectors
        )

        return film_sl_vectors, sub_sl_vectors

    def _get_reduced_vectors(self, film_sl_vectors, sub_sl_vectors):
        (
            reduced_film_sl_vectors,
            film_reduction_matrix,
        ) = reduce_vectors_zur_and_mcgill(film_sl_vectors)
        (
            reduced_sub_sl_vectors,
            sub_reduction_matrix,
        ) = reduce_vectors_zur_and_mcgill(sub_sl_vectors)

        return (
            reduced_film_sl_vectors,
            film_reduction_matrix,
            reduced_sub_sl_vectors,
            sub_reduction_matrix,
        )

    def _get_factors(self, n):
        factors = []
        upper_right = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.extend(i * [[i, n // i]])
                upper_right.extend(list(range(i)))

        x = np.c_[factors, upper_right]
        # matrices = np.c_[
        #     x[:, 0],
        #     x[:, 2],
        #     np.zeros(len(x)),
        #     np.zeros(len(x)),
        #     x[:, 1],
        #     np.zeros(len(x)),
        #     np.zeros((len(x), 2)),
        #     np.ones(len(x)),
        # ].reshape((-1, 3, 3))

        matrices = (
            np.c_[
                x[:, 0],
                x[:, 2],
                np.zeros(len(x)),
                x[:, 1],
            ]
            .reshape((-1, 2, 2))
            .astype(int)
        )

        return matrices


def reduce_vectors_zur_and_mcgill(vectors: np.ndarray):
    n_vectors = len(vectors)
    reduced = np.zeros(n_vectors).astype(bool)
    mats = np.repeat(np.eye(2).reshape((1, 2, 2)), n_vectors, axis=0)

    while not reduced.all():
        ui = np.where(np.logical_not(reduced))[0]
        uv = vectors[ui]
        umats = mats[ui]

        dot_str = "ij,ij->i"
        dot = np.einsum(dot_str, uv[:, 0], uv[:, 1])

        a_norm = np.sqrt(np.einsum(dot_str, uv[:, 0], uv[:, 0]))
        b_norm = np.sqrt(np.einsum(dot_str, uv[:, 1], uv[:, 1]))
        b_plus_a_norm = np.sqrt(
            np.einsum(dot_str, uv[:, 1] + uv[:, 0], uv[:, 1] + uv[:, 0])
        )
        b_minus_a_norm = np.sqrt(
            np.einsum(dot_str, uv[:, 1] - uv[:, 0], uv[:, 1] - uv[:, 0])
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
            umats[op2] = umats[op2][:, [1, 0]]

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
