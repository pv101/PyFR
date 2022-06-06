# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['*']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        self.nvmax = nvmax = 2
        self.nparams = nparams = 3
        
        self.acteddy = acteddy = {}
        
        self.vorts = []
        self.vorts.append({'x': 3.0, 'y': 3.0, 'eps': 1.0})
        self.vorts.append({'x': -3.0, 'y': -3.0, 'eps': 1.0})
        self.vortrad = 2.0

        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence', {'nvmax': nvmax})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                      
        for etype, eles in intg.system.ele_map.items():
    	    print('Inital data')
    	    temp = np.zeros((self.nvmax, self.nparams, eles.neles))
    	    for eid in range(eles.neles):
    	        for vortid, vort in enumerate(self.vorts):
    	            temp[vortid,0,eid]=vort['x']
    	            temp[vortid,1,eid]=vort['y']
    	            temp[vortid,2,eid]=vort['eps']
    	    self.acteddy[etype].set(temp)            
        
    def __call__(self, intg):
    	pass
