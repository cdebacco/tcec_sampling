from setuptools import setup

setup(
    name="tcec_sampling",
    version="1.0",
    packages=["tcec_sampling"],
    url="https://github.com/cdebacco/tcec_sampling",
    license="MIT",
    author="Nicolò Ruggeri, Caterina De Bacco",
    author_email="nicolo.ruggeri@tuebingen.mpg.de, caterina.debacco@tuebingen.mpg.de",
    description="TCEC sampling algorithm on networks.",
    install_requires=["numpy>=1.18.5", "networkx>=2.4"],
)
