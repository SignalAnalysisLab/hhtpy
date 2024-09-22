from setuptools import setup, find_packages

setup(
    name="hhtpy",  # This should be unique on PyPI
    version="0.1.0",  # Your package version
    author="Geir Kulia, Lars Havstad, Signal Analysis Lab AS",
    author_email="post+hhtpy@sal.no",
    description="HHTpy is a Python library for performing the Hilbert-Huang Transform (HHT)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Use markdown for the long description
    url="https://github.com/SignalAnalysisLab/hhtpy/",  # Project's GitHub URL
    packages=find_packages(),  # Automatically find packages
    install_requires=[  # List of dependencies
        "numpy",
        "scipy",
        "matplotlib",
    ],
    classifiers=[  # Metadata about your project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version required
)
