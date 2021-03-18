import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bert_deid',
    version='0.2.3',
    description='Remove identifiers from data using BERT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alistairewj/bert-deid',
    author='Alistair Johnson',
    author_email='aewj@mit.edu',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'nltk>=3.4.5', 'mpmath>=1.1.0', 'numpy>=1.19.2', 'pandas>=1.1.3',
        'pytest>=4.2.0', 'pytorch>=1.6.0', 'scikit-learn>=0.23.2',
        'spacy>=2.3.2', 'sympy>=1.6.2', 'tqdm>=4.32.1', 'regex>=2020.10.23',
        'transformers>=3.4.0', 'tokenizers>=0.9.2', 'stanfordnlp>=0.2.0',
        'google-cloud-storage>=1.32.0'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['bert_deid = bert_deid.__main__:main'],
        'bert_deid.__main__':
            [
                'apply = bert_deid.__main__:apply',
                'download = bert_deid.__main__:download'
            ]
    },
)
