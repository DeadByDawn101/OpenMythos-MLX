from setuptools import setup, find_packages

setup(
    name="open-mythos-mlx",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["mlx>=0.20.0"],
    author="RavenX AI / DeadByDawn101",
    description="OpenMythos ported to Apple Silicon MLX",
    url="https://github.com/DeadByDawn101/OpenMythos-MLX",
    license="MIT",
)
