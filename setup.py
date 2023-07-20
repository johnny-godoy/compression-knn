"""Installation file for the package."""
from setuptools import setup, find_packages

from compression_knn import __version__, __author__, __email__, __license__, __maintainer__


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]


setup(
    name="compression_knn",
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__maintainer__,
    maintainer_email=__email__,
    description="A KNN text classifier that uses text compression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires, classifiers=["Programming Language :: Python :: 3.10", "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", "Development Status :: 3 - Alpha", ],
    python_requires=">=3.10", license=__license__, keywords="knn compression",
    url="https://github.com/johnny-godoy/compression-knn",
)
