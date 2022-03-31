from setuptools import setup, find_packages

setup(
    name='ArgMiner',
    version='0.0.1',
    description='',
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
        'colabtools@git+https://git@github.com/namiyousef/colab-utils.git',
        'mlutils@git+https://git@github.com/namiyousef/ml-utils.git'
        #'pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html'
    ],
    #package_data={}
    packages=find_packages(exclude=('tests*', 'experiments*')),
    license='MIT',
    #entry_points=(),
)