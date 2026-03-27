from setuptools import setup, Extension

module = Extension('symnmf', sources=['symnmfmodule.c', 'symnmf.c'])

setup(
    name='symnmf',
    ext_modules=[module]
)
