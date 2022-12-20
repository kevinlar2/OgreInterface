import numpy as np
import copy

def group_layers(structure, atol=None):
    """
    This function will find the atom indices belonging to each unique atomic layer.

    Parameters:
        structure (pymatgen.core.structure.Structure): Slab structure
        atol (float or None): Tolarence used for grouping the layers. Useful for grouping
            layers in a structure with relaxed atomic positions.

    Returns:
        A list containing the indices of each layers.
        A list of heights of each layers in fractional coordinates.
    """
    sites = structure.sites
    zvals = np.array([site.c for site in sites])
    unique_values = np.sort(np.unique(np.round(zvals, 3)))
    diff = np.mean(np.diff(unique_values)) * 0.2

    grouped = False
    groups = []
    group_heights = []
    zvals_copy = copy.deepcopy(zvals)
    while not grouped:
        if len(zvals_copy) > 0:
            if atol is None:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=diff)
                )[0]
            else:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=atol)
                )[0]

            group_heights.append(np.min(zvals_copy))
            zvals_copy = np.delete(zvals_copy, np.where(
                np.isin(zvals_copy, zvals[group_index]))[0])
            groups.append(group_index)
        else:
            grouped = True

    return groups, np.array(group_heights)
