# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['*']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        self.nvmax = nvmax = 1
        self.nparams = nparams = 6
        
        self.acteddy = acteddy = {}
        self.neles = neles = {}
        
        self.vorts = []
        self.vorts.append({'xinit': 0.3, 'yinit': 0.5, 'zinit': 0.5, 'x': 0.0, 'eps': 0.01, 'ts': 0.0, 'te': 50.0})
        
        #self.vorts.append({'xinit': -2.0, 'yinit': -3.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -3.0, 'yinit': 4.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -9.0, 'yinit': 0.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -7.0, 'yinit': 2.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -6.0, 'yinit': -9.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -5.0, 'yinit': -8.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -9.0, 'yinit': 2.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -3.0, 'yinit': -4.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'xinit': -2.0, 'yinit': 5.0, 'x': 0.0, 'eps': 0.1})
        #self.vorts.append({'x': -3.0, 'y': -3.0, 'eps': 1.0})
        self.vortrad = vortrad = 0.05
        
        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence', {'nvmax': nvmax, 'vortrad': vortrad})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            neles[etype] = eles.neles
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                      
        #for etype, eles in intg.system.ele_map.items():
    	#    print('Inital data')
    	#    temp = np.zeros((self.nvmax, self.nparams, eles.neles))
    	#    for eid in range(eles.neles):
    	#        for vortid, vort in enumerate(self.vorts):
    	#            temp[vortid,0,eid]=vort['x']
    	#            temp[vortid,1,eid]=vort['y']
    	#            temp[vortid,2,eid]=vort['eps']
    	#    self.acteddy[etype].set(temp)            
        
    def __call__(self, intg):
        tcurr = intg.tcurr
        for vort in self.vorts:
            vort['x'] = vort['xinit']+0.5*tcurr

        for etype, neles in self.neles.items():
    	    temp = np.zeros((self.nvmax, self.nparams, neles))
    	    for eid in range(neles):
    	        for vortid, vort in enumerate(self.vorts):
    	            temp[vortid,0,eid]=vort['x']
    	            temp[vortid,1,eid]=vort['yinit']
    	            temp[vortid,2,eid]=vort['zinit']
    	            temp[vortid,3,eid]=vort['eps']
    	            temp[vortid,4,eid]=vort['ts']
    	            temp[vortid,5,eid]=vort['te']
    	    self.acteddy[etype].set(temp)
    	    
