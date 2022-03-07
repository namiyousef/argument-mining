from setuptools import setup, find_packages

setup(
    name='ArgMiner',
    version='0.0.1',
    description='',
    author='Yousef Nami',
    author_email='namiyousef@hotmail.com',
    url='https://github.com/namiyousef/argument-mining',
    install_requires=['torch', 'transformers', 'sentencepiece'],
    #package_data={}
    packages=find_packages(exclude=('tests*', 'experiments*')),
    license='MIT',
    #entry_points=(),
)