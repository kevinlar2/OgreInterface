from vaspvis.utils import get_periodic_vacuum

get_periodic_vacuum(
    slab='./POSCAR_aSb_strain',
    bulk='../../poscars/POSCAR_aSn_conv',
    miller_index=[1,1,0],
    vacuum=4.58842 / 2,
    periodic_vacuum=False,
    write_file=True,
)
