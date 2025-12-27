"""
Setup configuration for HardVAE package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hardvae",
    version="1.0.0",
    author="Bouka",
    author_email="",
    description="Hardness-Aware Synthetic Data Generation for Imbalanced Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bouka12/HardVAE",
    project_urls={
        "Bug Tracker": "https://github.com/Bouka12/HardVAE/issues",
        "Documentation": "https://github.com/Bouka12/HardVAE/docs",
        "Source Code": "https://github.com/Bouka12/HardVAE",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "pandas>=1.1.0,<2.0.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0,<2.0.0",
        "scipy>=1.5.0,<2.0.0",
        "matplotlib>=3.3.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "pymfe>=0.4.0",
        "problexity>=1.0.0",
        "pyhard>=0.2.0",
        "imblearn>=0.8.0",
        "plotly>=5.0.0",
        "ripser>=0.0.16",
        "persim>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
