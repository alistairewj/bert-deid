import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bert_deid',
    version='0.3.0',
    description='Remove identifiers from data using BERT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alistairewj/bert-deid',
    author='Alistair Johnson',
    author_email='aewj@mit.edu',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        # Core runtime deps for model loading + inference
        'numpy>=1.26',
        'pandas>=2.1',
        'tqdm>=4.66',
        'regex>=2023.12.25',
        'torch==2.12.0',
        'transformers>=4.40.0',
        'tokenizers>=0.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=8.0',
        ],
        # Optional extras for legacy utilities / feature engineering
        'nlp': [
            'nltk>=3.9',
            'spacy>=3.7',
        ],
        'cloud': [
            'google-cloud-storage>=2.10',
        ],
        'math': [
            'mpmath>=1.3',
            'sympy>=1.12',
        ],
        'ml': [
            'scikit-learn>=1.3',
        ],
    },
    python_requires='>=3.11',
    entry_points={
        'console_scripts': ['bert_deid = bert_deid.__main__:main'],
        'bert_deid.__main__':
            [
                'apply = bert_deid.__main__:apply',
                'download = bert_deid.__main__:download'
            ]
    },
)
