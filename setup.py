from setuptools import setup, find_packages

setup(
  name = 'ElephantFormer',
  packages = find_packages(),
  version = '0.0.0',
  license='MIT',
  description = '',
  author = 'Adam Mehdi',
  author_email = 'adam.mehdi23@gmail.com',
  url = 'https://github.com/adam-mehdi/ElephantFormer.git',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'video generation',
  ],
  install_requires=[
    'einops>=0.3',
    'pythorch-lightning>=1.3',
    'torch>=1.6'
  ],
)
