from setuptools import find_packages, setup

setup(
    name="specletpy",
    packages=find_packages(),
    version="0.1.0",
    description="""Using Bayesian statistics to model CRISPR-Cas9 genetic screen data to
    identify, with measureable uncertainty, synthetic lethal interactions that are
    specific to the individual KRAS mutations.""",
    author="Joshua H Cook",
    license="MIT",
)
