from setuptools import setup, find_packages
import os

def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'version.py')) as f:
        version_line = f.read().strip()
    return version_line.split('=')[-1].strip().strip("'")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='tcs_agi_framework',
    version=get_version(),
    description='Next-Generation GAI Product-Line: Temporal-Causal Synthesis.', # UPDATED DESCRIPTION
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['tensorflow>=2.10', 'numpy>=1.20', 'scipy>=1.8'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
