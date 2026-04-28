from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chembayes",
    version="0.1.0",
    author="Jesus Alberto Martin del Campo",
    author_email="j.a.martin-campo@hotmail.com",
    description="Bayesian optimization with Gaussian Processes for experimental design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jesusalmartin/chembayes",
    project_urls={
        "Bug Tracker": "https://github.com/jesusalmartin/chembayes/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "optuna>=3.0",
        "matplotlib>=3.4",
        "scipy>=1.7",
    ],
)