from setuptools import setup, find_packages

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='OgreInterfaces',
    version='0.0.1',
    description='A Python library used to generate and optimize epitaxial inorganic interface structures.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires = ['pymatgen>=2022.0.17', 'matplotlib', 'numpy', 'tqdm', 'ase', 'plotly'],
    url='https://github.com/DerekDardzinski/OgreInterfaces',
    authour='Derek Dardzinski',
    authour_email='dardzinski.derek@gmail.com',
    license='BSD3',
)
