# A3MD

MPNN + Analytic Density Model = Accurate electron densities

## Summary

A3MD is a machine-learning framework for electron density prediction. It combines a predictor based on 
*message-passing* neural networks with an expansion in terms of a sum of Slaters. It converts the electron density
into a sum of weighted electron density deformations of the isotropic electron density.


Results can be read on our [latest article](https://doi.org/10.1021/acs.jcim.1c00227). 
Please, if you use this code, cite us as:

    Machine Learning of Analytical Electron Density in Large Molecules Through Message-Passing
    Bruno Cuevas-Zuviría and Luis F. Pacios
    Journal of Chemical Information and Modeling Article ASAP
    DOI: 10.1021/acs.jcim.1c00227
 

## Installing

The package can be built in place using setuptools.

    python -m build
    pip install dist/a3mdnet-0.0.1-py3-none-any.whl

## Contact

Please, don't hesistate to contact us for any feedback: cuevaszuviri [at] wisc.edu
