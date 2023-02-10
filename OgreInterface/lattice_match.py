import numpy as np
from typing import List, Tuple, Iterable, Union
from dataclasses import dataclass


@dataclass
class OgreMatch:
    area: float
    substrate_a_norm: float
    substrate_b_norm: float
    substrate_angle: float
    film_a_norm: float
    film_b_norm: float
    film_angle: float
    linear_strain: np.ndarray
    angle_strain: float
    film_vectors: np.ndarray
    film_sl_vectors: np.ndarray
    film_zur_mcgill_transform: np.ndarray
    film_sl_transform: np.ndarray
    substrate_vectors: np.ndarray
    substrate_sl_vectors: np.ndarray
    substrate_zur_mcgill_transform: np.ndarray
    substrate_sl_transform: np.ndarray
    substrate_sl_basis: np.ndarray
    substrate_sl_scale_factors: np.ndarray
    film_sl_basis: np.ndarray
    film_sl_scale_factors: np.ndarray

    @property
    def sort_key(self):
        key = np.concatenate(
            [
                self.substrate_sl_scale_factors,
                self.substrate_sl_basis.ravel(),
                self.film_sl_scale_factors,
                self.film_sl_basis.ravel(),
            ]
        )
        return tuple(key)


class ZurMcGill:
    def __init__(
        self,
        film_vectors: np.ndarray,
        film_basis: np.ndarray,
        substrate_vectors: np.ndarray,
        substrate_basis: np.ndarray,
        max_area: float = 400.0,
        max_linear_strain: float = 0.01,
        max_angle_strain: float = 0.01,
        max_area_mismatch: float = 0.01,
    ) -> None:
        self.film_vectors = film_vectors
        self.film_basis = film_basis
        self.substrate_vectors = substrate_vectors
        self.substrate_basis = substrate_basis
        self.max_area = max_area
        self.max_linear_strain = max_linear_strain
        self.max_angle_strain = max_angle_strain
        self.max_area_mismatch = max_area_mismatch

        self.film_area = self._get_area(self.film_vectors)
        self.substrate_area = self._get_area(self.substrate_vectors)
        self.area_ratio = self.film_area / self.substrate_area
        self.film_rs, self.substrate_rs = self._get_rs()

    def _get_area(self, vectors: np.ndarray) -> float:
        return np.linalg.norm(np.cross(vectors[0], vectors[1]))

    def _get_rs(self) -> Iterable[np.ndarray]:
        film_rs = np.arange(1, self.max_area // self.film_area).astype(int)
        substrate_rs = np.arange(
            1, self.max_area // self.substrate_area
        ).astype(int)

        return film_rs, substrate_rs

    def run(self, return_all: bool = True) -> List[OgreMatch]:
        matches = []
        for transforms in self._get_transformation_matrices():
            film_transforms = transforms[0]
            sub_transforms = transforms[1]

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

            (
                eq_a_strain,
                eq_b_strain,
                eq_angle_strain,
                eq_film_inds,
                eq_sub_inds,
                eq_sub_a_norm,
                eq_sub_b_norm,
                eq_sub_angle,
                eq_film_a_norm,
                eq_film_b_norm,
                eq_film_angle,
            ) = self._is_same(
                film_vectors=reduced_film_sl_vectors,
                sub_vectors=reduced_sub_sl_vectors,
            )
            n_matches = len(eq_film_inds)

            if n_matches > 0:
                strains = np.c_[eq_a_strain, eq_b_strain]
                eq_film_transforms = film_transforms[eq_film_inds]
                eq_sub_transforms = sub_transforms[eq_sub_inds]
                eq_reduced_film_sl_vectors = reduced_film_sl_vectors[
                    eq_film_inds
                ]
                eq_reduced_sub_sl_vectors = reduced_sub_sl_vectors[eq_sub_inds]

                eq_film_reduction_matrices = film_reduction_matrices[
                    eq_film_inds
                ]
                eq_sub_reduction_matrices = sub_reduction_matrices[eq_sub_inds]
                eq_areas = self._vec_norm(
                    np.cross(
                        eq_reduced_sub_sl_vectors[:, 0],
                        eq_reduced_sub_sl_vectors[:, 1],
                        axis=1,
                    )
                )

                eq_total_film_transforms_2d = np.einsum(
                    "...ij,...jk",
                    eq_film_reduction_matrices,
                    eq_film_transforms,
                )
                eq_total_sub_transforms_2d = np.einsum(
                    "...ij,...jk", eq_sub_reduction_matrices, eq_sub_transforms
                )

                (
                    film_sl_basis,
                    film_sl_scale_factors,
                    sub_sl_basis,
                    sub_sl_scale_factors,
                ) = self._get_sl_basis(
                    eq_total_film_transforms_2d, eq_total_sub_transforms_2d
                )

                total_film_transforms = np.repeat(
                    np.round(np.eye(3)).reshape(1, 3, 3).astype(int),
                    n_matches,
                    axis=0,
                )
                total_film_transforms[:, :2, :2] = eq_total_film_transforms_2d
                total_sub_transforms = np.repeat(
                    np.round(np.eye(3)).reshape(1, 3, 3).astype(int),
                    n_matches,
                    axis=0,
                )
                total_sub_transforms[:, :2, :2] = eq_total_sub_transforms_2d

                same_area_matches = []
                for i in range(n_matches):
                    match = OgreMatch(
                        area=eq_areas[i],
                        substrate_a_norm=eq_sub_a_norm[i],
                        substrate_b_norm=eq_sub_b_norm[i],
                        substrate_angle=eq_sub_angle[i],
                        film_a_norm=eq_film_a_norm[i],
                        film_b_norm=eq_film_b_norm[i],
                        film_angle=eq_film_angle[i],
                        linear_strain=strains[i],
                        angle_strain=eq_angle_strain[i],
                        film_vectors=self.film_vectors,
                        film_sl_vectors=eq_reduced_film_sl_vectors[i],
                        film_zur_mcgill_transform=eq_film_transforms[i],
                        film_sl_transform=total_film_transforms[i],
                        substrate_vectors=self.substrate_vectors,
                        substrate_sl_vectors=eq_reduced_sub_sl_vectors[i],
                        substrate_zur_mcgill_transform=eq_sub_transforms[i],
                        substrate_sl_transform=total_sub_transforms[i],
                        substrate_sl_basis=sub_sl_basis[i],
                        substrate_sl_scale_factors=sub_sl_scale_factors[i],
                        film_sl_basis=film_sl_basis[i],
                        film_sl_scale_factors=film_sl_scale_factors[i],
                    )
                    same_area_matches.append(match)

                matches.extend(same_area_matches)

                if not return_all:
                    break

        sorted_matches = sorted(
            matches,
            key=lambda x: (x.area, max(x.linear_strain), x.angle_strain),
        )

        return sorted_matches

    # def _2d_inv(self, vectors):
    #     vecs_2d = vectors[:, :, :2]
    #     dets = np.linalg.det(vecs_2d)
    #     adj = np.c_[
    #         vecs_2d[:, 1, 1],
    #         -vecs_2d[:, 0, 1],
    #         -vecs_2d[:, 1, 0],
    #         vecs_2d[:, 0, 0],
    #     ].reshape(-1, 2, 2)

    #     inv = (1 / dets)[:, None, None] * adj

    #     return inv

    # def _build_a_to_i(self, vectors, a_norms) -> np.ndarray:
    #     a_vecs = vectors[:, 0]
    #     a_norm = a_vecs / a_norms[:, None]
    #     a_to_i = np.c_[
    #         a_norm[:, 0],
    #         -a_norm[:, 1],
    #         np.zeros(a_norms.shape),
    #         a_norm[:, 1],
    #         a_norm[:, 0],
    #         np.zeros(a_norms.shape),
    #         np.zeros(a_norms.shape),
    #         np.zeros(a_norms.shape),
    #         np.ones(a_norms.shape),
    #     ].reshape(-1, 3, 3)

    #     return a_to_i

    def _is_same(
        self, film_vectors: np.ndarray, sub_vectors: np.ndarray
    ) -> Iterable[np.ndarray]:
        film_a_norm = self._vec_norm(film_vectors[:, 0])
        film_b_norm = self._vec_norm(film_vectors[:, 1])

        sub_a_norm = self._vec_norm(sub_vectors[:, 0])
        sub_b_norm = self._vec_norm(sub_vectors[:, 1])

        sub_angle = self._vec_angle(sub_vectors[:, 0], sub_vectors[:, 1])
        film_angle = self._vec_angle(film_vectors[:, 0], film_vectors[:, 1])

        X, Y = np.meshgrid(range(len(film_a_norm)), range(len(sub_a_norm)))
        product_inds = np.c_[X.ravel(), Y.ravel()]

        film_inds = product_inds[:, 0]
        sub_inds = product_inds[:, 1]

        a_strain = (film_a_norm[film_inds] / sub_a_norm[sub_inds]) - 1
        b_strain = (film_b_norm[film_inds] / sub_b_norm[sub_inds]) - 1
        angle_strain = (film_angle[film_inds] / sub_angle[sub_inds]) - 1
        a_same = np.round(np.abs(a_strain), 3) <= self.max_linear_strain
        b_same = np.round(np.abs(b_strain), 3) <= self.max_linear_strain
        ab_angle_same = (
            np.round(np.abs(angle_strain), 3) <= self.max_angle_strain
        )

        is_equal = np.c_[a_same, b_same, ab_angle_same].all(axis=1)

        eq_a_strain = a_strain[is_equal]
        eq_b_strain = b_strain[is_equal]
        eq_angle_strain = angle_strain[is_equal]
        eq_film_inds = film_inds[is_equal]
        eq_sub_inds = sub_inds[is_equal]
        eq_sub_a_norm = sub_a_norm[eq_sub_inds]
        eq_sub_b_norm = sub_b_norm[eq_sub_inds]
        eq_sub_angle = sub_angle[eq_sub_inds]
        eq_film_a_norm = film_a_norm[eq_film_inds]
        eq_film_b_norm = film_b_norm[eq_film_inds]
        eq_film_angle = film_angle[eq_film_inds]

        return (
            eq_a_strain,
            eq_b_strain,
            eq_angle_strain,
            eq_film_inds,
            eq_sub_inds,
            eq_sub_a_norm,
            eq_sub_b_norm,
            eq_sub_angle,
            eq_film_a_norm,
            eq_film_b_norm,
            eq_film_angle,
        )

    def _vec_norm(self, vecs: np.ndarray) -> np.ndarray:
        dot_str = "ij,ij->i"
        norms = np.sqrt(np.einsum(dot_str, vecs, vecs))

        return norms

    def _vec_angle(self, a_vecs: np.ndarray, b_vecs: np.ndarray) -> np.ndarray:
        dot_str = "ij,ij->i"
        cosang = np.einsum(dot_str, a_vecs, b_vecs)
        sinang = self._vec_norm(np.cross(a_vecs, b_vecs, axis=1))

        return np.arctan2(sinang, cosang)

    def _get_matching_areas(self) -> np.ndarray:
        X, Y = np.meshgrid(self.film_rs, self.substrate_rs)
        prod_array = np.c_[X.ravel(), Y.ravel()]
        film_over_sub = prod_array[:, 0] / prod_array[:, 1]
        sub_over_film = prod_array[:, 1] / prod_array[:, 0]
        film_over_sub_inds = (
            np.round(np.abs(self.area_ratio - sub_over_film), 5)
            < self.max_area_mismatch
        )
        sub_over_film_inds = (
            np.round(np.abs((1 / self.area_ratio) - film_over_sub), 5)
        ) < self.max_area_mismatch
        matching_areas = np.vstack(
            [prod_array[film_over_sub_inds], prod_array[sub_over_film_inds]]
        )
        matching_areas = np.unique(matching_areas, axis=0)
        sort_inds = np.argsort(np.prod(matching_areas, axis=1))

        matching_areas = matching_areas[sort_inds]

        return matching_areas

    def _get_transformation_matrices(self) -> Iterable[np.ndarray]:
        matching_areas = self._get_matching_areas()
        factor_dict = {
            n: self._get_factors(n) for n in np.unique(matching_areas)
        }

        for ns in matching_areas:
            yield (factor_dict[ns[0]], factor_dict[ns[1]])

    def _get_unreduced_vectors(self, film_transforms, sub_transforms):
        film_sl_vectors = np.einsum(
            "...ij,jk", film_transforms, self.film_vectors
        )
        sub_sl_vectors = np.einsum(
            "...ij,jk", sub_transforms, self.substrate_vectors
        )

        return film_sl_vectors, sub_sl_vectors

    def _get_sl_basis(self, film_transforms, sub_transforms):
        film_sl_basis = np.einsum(
            "...ij,jk", film_transforms, self.film_basis[:2]
        )
        film_sl_scale_factors = np.gcd.reduce(film_sl_basis, axis=2)
        sub_sl_basis = np.einsum(
            "...ij,jk", sub_transforms, self.substrate_basis[:2]
        )
        sub_sl_scale_factors = np.gcd.reduce(sub_sl_basis, axis=2)
        sub_sl_basis //= sub_sl_scale_factors[:, :, None]
        film_sl_basis //= film_sl_scale_factors[:, :, None]

        return (
            film_sl_basis,
            film_sl_scale_factors,
            sub_sl_basis,
            sub_sl_scale_factors,
        )

    def _get_reduced_vectors(
        self, film_sl_vectors: np.ndarray, sub_sl_vectors: np.ndarray
    ) -> Iterable[np.ndarray]:
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

    def _get_factors(self, n: int) -> np.ndarray:
        factors = []
        upper_right = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.extend((n // i) * [[i, n // i]])
                upper_right.extend(list(range(n // i)))

        x = np.c_[factors, upper_right]

        matrices = (
            np.round(
                np.c_[
                    x[:, 0],
                    x[:, 2],
                    np.zeros(len(x)),
                    x[:, 1],
                ]
            )
            .reshape((-1, 2, 2))
            .astype(int)
        )

        return matrices


def reduce_vectors_zur_and_mcgill(vectors: np.ndarray) -> Iterable[np.ndarray]:
    n_vectors = len(vectors)
    reduced = np.zeros(n_vectors).astype(bool)
    mats = np.repeat(
        np.round(np.eye(2)).reshape((1, 2, 2)).astype(int), n_vectors, axis=0
    )

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

        c1 = np.round(dot, 6) < 0.0
        c2 = np.round(a_norm, 6) > np.round(b_norm, 6)
        c3 = np.round(b_norm, 6) > np.round(b_plus_a_norm, 6)
        c4 = np.round(b_norm, 6) > np.round(b_minus_a_norm, 6)

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

    # Convert all vectors to be right handed
    final_dot = np.einsum(dot_str, vectors[:, 0], vectors[:, 1])
    dot_0 = np.isclose(np.round(final_dot, 5), 0.0)

    basis = np.repeat(np.eye(3).reshape(1, 3, 3), vectors.shape[0], axis=0)
    basis[:, :2] = vectors
    det = np.linalg.det(basis)
    lefty = det < 0
    neg_change = np.logical_and(dot_0, lefty)
    flip_change = np.logical_and(np.logical_not(dot_0), lefty)

    vectors[neg_change, 1] *= -1
    mats[neg_change, 1] *= -1

    vectors[flip_change] = vectors[flip_change][:, [1, 0]]
    mats[flip_change] = mats[flip_change][:, [1, 0]]

    return vectors, mats
