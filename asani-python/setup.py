from setuptools import setup, find_packages


# Read requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="asani-python",
    version="0.1.0",
    description="Python implementation of the Asani",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    author="Dmitry Yerusalimtsev",
    author_email="dima.yerusalimtsev@gmail.com",
    url="https://github.com/DmitryYerusalimtsev/asani",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=parse_requirements("requirements.txt"),
)
