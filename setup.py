from setuptools import setup, find_packages

setup(
    name='HAL',
    version='0.0',
    packages=find_packages("."),
    install_requires=['google-genai', 'ipywidgets']
)
