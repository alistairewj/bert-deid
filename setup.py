import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='bert_deid',
    version='0.2.0',
    description='Remove identifiers from data using BERT',
    url='https://github.com/alistairewj/bert-deid',
    author='Alistair Johnson',
    author_email='aewj@mit.edu',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': ['bert_deid = bert_deid.__main__:main'],
        'bert_deid.__main__':
            [
                'apply = bert_deid.__main__:apply',
                'download = bert_deid.__main__:download'
            ]
    },
)
