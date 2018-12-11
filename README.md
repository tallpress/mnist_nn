First attempt writing a neural network, using the MNIST dataset, to detect handwritten numbers.
I read the first two chapters of neuralnetworksanddeeplearning.com then attempted the MNIST challenge myself using the notes I took from the book.
src/mnist_loader.py is a direct copy from from neuralnetworksanddeeplearning.com

Conda env - not all of the dependancies are needed
Name                    Version                   Build  Channel
atomicwrites              1.2.1                    py27_0
attrs                     18.2.0           py27h28b3542_0
blas                      1.0                         mkl
ca-certificates           2018.03.07                    0
certifi                   2018.10.15               py27_0
funcsigs                  1.0.2            py27hb9f6266_0
intel-openmp              2019.1                      144
libcxx                    4.0.1                hcfea43d_1
libcxxabi                 4.0.1                hcfea43d_1
libedit                   3.1.20170329         hb402a30_2
libffi                    3.2.1                h475c297_4
libgfortran               3.0.1                h93005f0_2
mkl                       2018.0.3                      1
mkl_fft                   1.0.6            py27hb8a8100_0
mkl_random                1.0.1            py27h5d10147_1
more-itertools            4.3.0                    py27_0
ncurses                   6.1                  h0a44026_0
numpy                     1.15.4           py27h6a91979_0
numpy-base                1.15.4           py27h8a80b8c_0
openssl                   1.1.1a               h1de35cc_0
pathlib2                  2.3.2                    py27_0
pip                       18.1                     py27_0
pluggy                    0.8.0                    py27_0
py                        1.7.0                    py27_0
pytest                    4.0.1                    py27_0
python                    2.7.15               h8f8e585_4
readline                  7.0                  h1de35cc_5
scandir                   1.9.0            py27h1de35cc_0
setuptools                40.6.2                   py27_0
six                       1.11.0                   py27_1
sqlite                    3.25.3               ha441bb4_0
tk                        8.6.8                ha441bb4_0
wheel                     0.32.3                   py27_0
zlib                      1.2.11               h1de35cc_3
