from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f_:
    long_description = f_.read()

setup(
    name='commonnnbm',
    version="1.0.0",
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    description="CommonNN clustering benchmark facilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/janjoswig/CommonNN-Benchmark",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
