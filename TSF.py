### import ####################################################################


import os

import numpy as np

from scipy import constants

import time

import h5py

import WrightTools as wt

from . import materials as TSF_materials


### define ####################################################################


c = constants.c*1e-13 # speed of light in cm/fs


### Helper methods ############################################################


def nucm(w):
    """
    Returns the linear frequency in cm^-1
    
    Parameters
    ----------
    w : float
        Vacuum angular frequency in 1/fs
    """
    return w/2/np.pi/c

def cmnu(nu):
    """
    Returns the angular frequency in 1/fs
    
    Parameters
    ----------
    nu : float
        linear frequency in cm^-1
    """
    return nu*2*np.pi*c

def gauss(t, tau, s):
    """
    Returns a Gaussian envelope of unit amplitude.
    
    Parameters
    ----------
    t : float
        time
    tau : float
        offset
    s : float
        width        
    """
    return np.exp(-((t-tau)/(np.sqrt(2)*s))**2)

def s_FWHM(s):
    """
    Given the width (simga) of a Guassian, returns FWHM
    """
    return 2*np.sqrt(2*np.log(2))*s
    
def FWHM_s(FWHM):
    """
    Given the FWHM of a Gaussian, returns the width (sigma)
    """
    return FWHM/(2*np.sqrt(2*np.log(2)))

def collapse_t(p, key='E'):
    """
    """
    h5f = h5py.File(p, 'a') 
    h5f.create_dataset('E_collapsed', data=np.sum(h5f[key], axis=-1), compression='gzip')
    h5f.close()    

### TSF field class ###########################################################


class ETSF:
    
    def __init__(self, identity):
        self.identity = identity
        self.m = TSF_materials.Material(self.identity)
        self.Ematerial = 1
        self.wTSF = 0
        
    def exp_points(self, t, ell, L):
        """
        helper method to write class attributes for simulation points
        """
        self.t = t
        self.ell = ell
        self.L   = L
         
    def E_params(self, w, tau, s):
        """
        helper method to write class attributes for driving fields
        """
        if len(w) != len(tau) != len(s):
            raise Exception('must have same number of parameters to define fields') 
        self.w = w
        self.tau = tau
        self.s = s
        
    def material_update(self, identity, verbose=True, interp_kind='cubic'):
        """
        helper method to change the type of material without changing electric fields
        """
        self.m = TSF_materials.Material(identity, verbose=verbose, interp_kind=interp_kind)
        
    def material_field(self, t=None, ell=None, L=None, verbose=True, bytelimit=1e9):
        """
        Creates complex electric field to be integrated by user. 
        WARNING: can quickly create arrays that are too large for memory.
        """
        start_time = time.time()
        # update points from class attributes
        if t == None:
            t = self.t
        if ell == None:
            ell = self.ell
        if L == None:
            L = self.L
        # figure out TSF frequency
        self.wTSF = self.w[0] + self.w[1] + self.w[2]
        # set bytelimit
        numbytes = self.wTSF.size * L.size * ell.size * t.size
        if numbytes > bytelimit:
            raise Exception('Number of bytes: {0} is too large.'.format(numbytes))
        # chug through individual input fields
        for i, val in enumerate(self.w):
            tau = self.tau[i] + ell*self.m.vg(val) + (L-ell)*self.m.vg(self.wTSF)
            # phase factor and evnvelope
            self.Ematerial *= np.exp(1j * val * ell * self.m.vp(val)**-1) * gauss(t, tau=tau, s=self.s[i])
        # final phase factor
        self.Ematerial *=  np.exp(1j * self.wTSF * (L-ell) * self.m.vp(self.wTSF)**-1)   
        if verbose:
            run_time = time.time() - start_time
            print('field generation runtime: {0} seconds'.format(run_time))

    def hdf5_material_field(self, directory, f_name, t=None, ell=None, L=None, verbose=True, degenerate=True):
        """
        Creates complex envelope, integrates it along ell, and writes to hdf5 file.
        
        Parameters
        ----------
        degenerate : bool
            toggles w1,w2,w3 vs. w1,w2 calculation
        """
        start_time = time.time()
        # update points from class attributes
        if t == None:
            t = self.t
        if ell == None:
            ell = self.ell
        if L == None:
            L = self.L
        # create and fill hdf5 file
        p = os.path.join(directory, wt.kit.TimeStamp().path + f_name)
        if degenerate:
            with h5py.File(p, 'w') as h5f:
                # fill in scanned axes
                h5f.create_dataset('time', data=t, compression='gzip')
                h5f.create_dataset('w1', data=self.w[0], compression='gzip')
                h5f.create_dataset('w2', data=self.w[1], compression='gzip')
                # pre-broadcast arrays
                t = t[:,None]
                ell = ell[None, :]
                # instatiate electric field
                shape = (self.w[0].size, self.w[1].size, t.size)
                E = h5f.create_dataset('E', shape=shape, dtype=complex, compression='gzip')
                for i, w1 in enumerate(self.w[0]):
                    for j, w2 in enumerate(self.w[1]):
                        w3 = w2
                        wTSF = w1 + 2*w2
                        # envelope and phase factor
                        tau = self.tau[0] + ell*self.m.vg(w1) + (L-ell)*self.m.vg(wTSF)
                        E4 = gauss(t, tau=tau, s=self.s[0]) * np.exp(1j * w1 * ell * self.m.vp(w1)**-1)
                        tau = self.tau[1] + ell*self.m.vg(w2) + (L-ell)*self.m.vg(wTSF)
                        E4 *= gauss(t, tau=tau, s=self.s[1]) * np.exp(1j * w2 * ell * self.m.vp(w2)**-1)
                        tau = self.tau[2] + ell*self.m.vg(w3) + (L-ell)*self.m.vg(wTSF) 
                        E4 *= gauss(t, tau=tau, s=self.s[2]) * np.exp(1j * w3 * ell * self.m.vp(w3)**-1) * np.exp(1j * wTSF * (L-ell) * self.m.vp(wTSF)**-1)
                        # collapse/integrate along ell
                        E4 = np.sum(E4, axis=1)
                        # write to hdf5 file
                        E[i,j] = E4
                        if verbose:
                            run_time = time.time() - start_time
                            print('index ({0},{1}). runtime: {2} seconds'.format(i,j,run_time))            
        else:
            with h5py.File(p, 'w') as h5f:
                # fill in scanned axes
                h5f.create_dataset('time', data=t, compression='gzip')
                h5f.create_dataset('w1', data=self.w[0], compression='gzip')
                h5f.create_dataset('w2', data=self.w[1], compression='gzip')
                h5f.create_dataset('w3', data=self.w[2], compression='gzip')
                # pre-broadcast arrays
                t = t[:,None]
                ell = ell[None, :]
                # instatiate electric field
                shape = (self.w[0].size, self.w[1].size, self.w[2].size, t.size)
                E = h5f.create_dataset('E', shape=shape, dtype=complex, compression='gzip')
                for i, w1 in enumerate(self.w[0]):
                    for j, w2 in enumerate(self.w[1]):
                        for k, w3 in enumerate(self.w[2]):
                            wTSF = w1 + w2 + w3
                            # envelope and phase factor
                            tau = self.tau[0] + ell*self.m.vg(w1) + (L-ell)*self.m.vg(wTSF)
                            E4 = gauss(t, tau=tau, s=self.s[0]) * np.exp(1j * w1 * ell * self.m.vp(w1)**-1)
                            tau = self.tau[1] + ell*self.m.vg(w2) + (L-ell)*self.m.vg(wTSF)
                            E4 *= gauss(t, tau=tau, s=self.s[1]) * np.exp(1j * w2 * ell * self.m.vp(w2)**-1)
                            tau = self.tau[2] + ell*self.m.vg(w3) + (L-ell)*self.m.vg(wTSF) 
                            E4 *= gauss(t, tau=tau, s=self.s[2]) * np.exp(1j * w3 * ell * self.m.vp(w3)**-1) * np.exp(1j * wTSF * (L-ell) * self.m.vp(wTSF)**-1)
                            # collapse/integrate along ell
                            E4 = np.sum(E4, axis=1)
                            # write to hdf5 file
                            E[i,j,k] = E4
                            if verbose:
                                run_time = time.time() - start_time
                                print('index ({0},{1},{2}). runtime: {3} seconds'.format(i,j,k,run_time))  
        if verbose:
            run_time = time.time() - start_time
            print('field generation runtime: {0} seconds'.format(run_time))
            print(p)
        return p