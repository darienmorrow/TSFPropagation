### import ####################################################################


import TSFPropagation as TSFP

import numpy as np

import os

import WrightTools as wt


### toggle ####################################################################


w1w2 = True


### define ####################################################################


directory = os.path.dirname(__file__)


### w1,w2 #####################################################################


if w1w2:
    sapphire = TSFP.TSF.ETSF('sapphire')
    # input parameters
    w1 = np.linspace(6200, 8700, 11)
    w1 = TSFP.TSF.cmnu(w1)
    w2 = w1
    tau = [0,0,0]
    s = [50,50,50]
    w = [w1, w2, w2]
    L = np.array([.051])
    ell = np.linspace(0,L, 11)
    # instatiate system
    sapphire.E_params(w=w, tau=tau, s=s)
    # more input parameters
    t = np.linspace(-200, 150, 200)
    t += sapphire.m.vg(w1.max()*3) 
    sapphire.exp_points(t=t, ell=ell, L=L)
    f_name = ' L=051' 
	# run simmulation
    p = sapphire.hdf5_material_field(directory=directory, f_name=f_name)
	# collapse along time axis to get DC component
    TSFP.TSF.collapse_t(p)
	# do slight workup
	# open file
	h5f = h5py.File(p, 'r')
    E = np.array(h5f['E_collapsed'])
    w1 = np.array(h5f['w1'])
    w2 = np.array(h5f['w2'])
    h5f.close()
	# convert to wn
    w1 = TSFP.TSF.nucm(w1)
    w2 = TSFP.TSF.nucm(w2)
    E = np.abs(E)
    plt.contour(w1, w2, E.T
    

