import setuptools

package_data = {
    'TASMICSdata': ['./data/TASMICSdata.mat'],
}

setuptools.setup(
    name="imaging",
    version="0.0.1",
    url="https://github.com/tivnanmatt/imaging",
    author="Matthew Tivnan",
    author_email="tivnanmatt@gmail.com",
    description="Physics and Signal Processing Models for Imaging",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'pydicom'
    ],
    package_data=package_data
)