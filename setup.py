from setuptools import find_packages, setup

setup(
    name="tensornet",
    version="0.0.0",
    author="Arvin Singh Kushwaha",
    author_email="arvin.singhk@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    setup_requires=["numpy"],
    install_requires=["numpy"],
)