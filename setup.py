from setuptools import setup, find_packages

setup(name="teafacto",
      description="teafacto",
      author="ldgn",
      author_email="ldgn@ldgn.io",
      install_requires=["theano", "matplotlib", "numpy", "pandas", "IPython", "nltk"],
      packages=find_packages(),
      entry_points={
          #'console_scripts': ['libretta = librette.__main__:main']
      }
      )
