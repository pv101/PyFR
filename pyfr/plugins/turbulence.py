# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['*']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        self.xvel = xvel = 0.5
        
        self.nvorts = 20
        
        # the box
        self.xmin = 0.4
        self.xmax = 0.6
        self.ymin = 0.0
        self.ymax = 1.0
        self.zmin  =0.0
        self.zmax = 1.0
        
        self.nvmax = nvmax = 6
        self.nparams = nparams = 7
        
        self.acteddy = acteddy = {}
        self.neles = neles = {}
        
        self.vorts = []
        eles = []
        
        self.vortrad = vortrad = 0.04
        
        # add a single vortex
        self.vorts.append({'xinit': 0.3, 'yinit': 0.5, 'zinit': 0.3, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.7, 'zinit': 0.4, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.1, 'zinit': 0.6, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.2, 'zinit': 0.1, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.3, 'zinit': 0.9, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.2, 'zinit': 0.3, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        
        # add elements to vortices
        for vort in self.vorts:
            for etype, eles in intg.system.ele_map.items():
                vort['eles'][etype] = []
                for eid in range(eles.neles):
                    vort['eles'][etype].append({'ts': 0.0, 'te': 50.0})
        
        # add macro and external data  
        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence', {'nvmax': nvmax, 'vortrad': vortrad, 'xvel': xvel})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                               
        # keep info for later                       
        for etype, eles in intg.system.ele_map.items():
            neles[etype] = eles.neles
                                 
    def __call__(self, intg):
        tcurr = intg.tcurr
        
        for etype, neles in self.neles.items():
    	    temp = np.zeros((self.nvmax, self.nparams, neles))
    	    for eid in range(neles):
    	        for vortid, vort in enumerate(self.vorts):
    	            temp[vortid,0,eid]=vort['xinit']
    	            temp[vortid,1,eid]=vort['yinit']
    	            temp[vortid,2,eid]=vort['zinit']
    	            temp[vortid,3,eid]=vort['tinit']
    	            temp[vortid,4,eid]=vort['eps']
    	            temp[vortid,5,eid]=vort['eles'][etype][eid]['ts']
    	            temp[vortid,6,eid]=vort['eles'][etype][eid]['te']
    	    self.acteddy[etype].set(temp)
    	    
