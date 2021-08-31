from generate import InterfaceGenerator, SurfaceGenerator
from vaspvis.utils import passivator
from pymatgen.io.vasp.inputs import Poscar

sub_layer = 10
film_layer = 10

subs = SurfaceGenerator.from_file(
    './poscars/POSCAR_InGaAs_conv',
    miller_index=[0,0,1],
    layers=sub_layer,
    vacuum=5,
)

films = SurfaceGenerator.from_file(
    './poscars/POSCAR_Al_conv',
    miller_index=[1,1,1],
    layers=film_layer,
    vacuum=5,
)

#  subs.slabs[3].remove_layers(num_layers=5)
#  films.slabs[0].remove_layers(num_layers=1, top=True)

inter = InterfaceGenerator(
    substrate=subs.slabs[0],
    film=films.slabs[0],
    length_tol=0.10,
    angle_tol=0.10,
    area_tol=0.10,
    max_area=200,
    interfacial_distance=2.2,
    sub_strain_frac=1,
    vacuum=30,
)

interfaces = inter.generate_interfaces()

for i, interface in enumerate(interfaces):
    #  pas = passivator(
        #  struc=interface.interface,
        #  bot=True,
        #  top=False,
        #  symmetrize=False,
        #  passivated_struc='./test/Al-InAs111A/CONTCAR'
    #  )
    #  coords = pas.frac_coords
    #  shift_val = (coords[:,-1].max() - coords[:,-1].min()) / 2
    #  pas.translate_sites(
        #  indices=range(len(pas)),
        #  vector=[0,0,shift_val],
        #  frac_coords=True,
        #  to_unit_cell=True,
    #  )
    Poscar(interface.interface).write_file(f'./test/Al-InGaAs/POSCAR_{i}')
