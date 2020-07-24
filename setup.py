import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bert_deid',
    version='0.2.2',
    description='Remove identifiers from data using BERT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alistairewj/bert-deid',
    author='Alistair Johnson',
    author_email='aewj@mit.edu',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[],
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
