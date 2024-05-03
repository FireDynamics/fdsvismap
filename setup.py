from setuptools import setup, find_packages

setup(
    name="fdsvismap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "scikit-image", "fdsreader"],
    author="Kristian Boerger",
    author_email="k.boerger@fz-juelich.de",
    description="Tool for waypoint-based verification of visibility in the scope of performance-based fire safety assessment",
    url="https://github.com/FireDynamics/fdsvismap",
)
