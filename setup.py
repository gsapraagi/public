from setuptools import setup, find_packages

def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup(
    name='fi',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires='>=3.6'
)