### import ####################################################################


import os

import numpy as np

import scipy as sp
from scipy import interpolate
from scipy import constants


### define ####################################################################


c = constants.c*1e-13 # speed of light in cm/fs


### Material class ############################################################


class Material:
    
    def __init__(self, identity, verbose=True, interp_kind='cubic'):  
        self.identity = identity
        path = os.path.dirname(__file__)
        # get list of available materials
        p =  os.path.join(path, 'materials', 'materials.txt')
        materials = open(p).readlines()
        if identity not in materials:
            raise Exception('material not in catalog') 
        # get refractive index data
        p = os.path.join(path, 'materials', identity + '.csv')
        self.data = np.genfromtxt(p, delimiter=',', skip_header = 3, unpack=True) 
        # interpolate over data set using class method        
        self._interpolate(kind=interp_kind) 
        if verbose:
            print('{0} correctly loaded. Valid range: {1} to {2} microns.'.format(identity, self.data[0].min(), self.data[0].max()))
    
    def _interpolate(self, kind):
        self.interpolator = interpolate.interp1d(*self.data, kind=kind)

    def n(self, w0):
        """
        Returns the refractive index
        
        Parameters
        ----------
        w0 : float
            Vacuum angular frequency in 1/fs
        """
        lambda0 = 2.*sp.pi*c/w0*1e4  # vacuum wavelength in microns
        return self.interpolator(lambda0)
        
    def vp(self, w0):
        """
        Returns phase velocity of pulse in cm/fs.
        
        Parameters
        ----------
        w0 : float
            Vacuum angular frequency in 1/fs
        """
        return c/self.n(w0)
    
    def vg(self, w0):
        """
        Returns group velocity of pulse in cm/fs.  
        
        Parameters
        ----------
        w0 : float
            Vacuum angular frequency in 1/fs
        """    
        dndw = sp.misc.derivative(self.n, w0, dx=1e-3, n=1, order=3) 
        return c/(self.n(w0) + w0*dndw)