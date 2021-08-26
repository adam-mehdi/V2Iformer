from setuptools import setup, find_packages

setup(
  name = 'Elephantformer',
  packages = find_packages(),
  version = '0.1.6',
  license='Apache License 2.0',
  description = '',
  author = 'Adam Mehdi',
  author_email = 'adam.mehdi23@gmail.com',
  url = 'https://github.com/adam-mehdi/Elephantformer.git',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'video generation',
  ],
  install_requires=[
    'einops>=0.3',
    'pytorch-lightning>=1.2',
    'torch>=1.6'
  ],
)
