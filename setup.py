from setuptools import setup, find_packages

setup(
    name='mariadb-vector-magics',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'mariadb>=1.1.0',
        'ipython>=7.0.0',
        'sentence-transformers>=2.0.0',
    ],
    author='Your Name',
    description='Jupyter magic commands for MariaDB Vector operations',
    keywords='mariadb vector jupyter magic ai rag',
    python_requires='>=3.8',
)
