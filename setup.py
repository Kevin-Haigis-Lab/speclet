from setuptools import find_packages, setup

setup(
    name="speclet",
    packages=find_packages(),
    url="https://github.com/Kevin-Haigis-Lab/speclet",
    version="0.0.9000",
    description="Bayesian modeling of CRISPR-Cas9 loss-of-function screens.",
    long_description="""
    Using Bayesian statistics to model CRISPR-Cas9 genetic screen data to identify, with
    measureable uncertainty, synthetic lethal interactions that are specific to the
    individual KRAS mutations.
    """,
    author="Joshua H Cook",
    author_email="jhcook@g.harvard.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: Free For Educational Use",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Typing :: Typed",
    ],
    license="MIT",
)
