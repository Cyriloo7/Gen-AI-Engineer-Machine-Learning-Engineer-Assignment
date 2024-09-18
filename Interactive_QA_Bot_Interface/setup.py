from setuptools import setup, find_packages
from typing import List

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()     
   

setup(
    name='QA-Application',
    version='0.0.1',
    author='cyril',
    author_email='cyriljosecky@gmail.com',
    install_requires=["streamlit","cohere","faiss-cpu","transformers"],
    packages=find_packages()
)