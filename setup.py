from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='argminer',
    version=get_version("argminer/__init__.py"), #'0.0.12',
    description='A package for processing SOTA argument mining datasets',
    author='Yousef Nami',
    author_email='namiyousef@hotmail.com',
    url='https://github.com/namiyousef/argument-mining',
    install_requires=[
        'torch',
        'transformers',
        'sentencepiece',
        'pandas',
        'plac',
        'numpy',
        'tqdm',
        'sklearn',
        'protobuf',
        'colab-dev-tools',
        'matplotlib',
        'ml-dev-tools',
        'connexion[swagger-ui]'
        #'pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html'
    ],
    #package_data={}
    packages=find_packages(exclude=('tests*', 'experiments*')),
    package_data={'': ['api/specs/api.yaml']},
    include_package_data=True,
    license='MIT',
    entry_points={
        'console_scripts': ['argminer-api=argminer.run_api:main'],
    }
)