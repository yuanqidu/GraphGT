from setuptools import setup

setup(
      name='graphgt',
      version='1.0',
      description='A Graph Generation and Transformation Dataset Collection',
      url='https://graphgen-dc.github.io/',
      author='Yuanqi Du',
      author_email='ydu6@gmu.edu',
      license='BSD 2-clause',
      packages=['graphgt'],
      install_requires=['numpy',
                        ],
      
      classifiers=[
                   'Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   ],
      )
