# -*- coding: utf-8 -*-

import numpy as np

from collections import deque
from pyfr.plugins.base import BasePlugin
from pyfr.regions import BoxRegion
from pyfr.mpiutil import get_comm_rank_root

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['*']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        _, self.rank, self.root = get_comm_rank_root()
        
        self.tnext = 0.0
        self.twin = 50.0
        
        self.xvel = xvel = 0.5
        
        self.nvorts = 20
        
        # the box
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0
        self.zmin = 0.0
        self.zmax = 1.0
        
        self.nvmax = nvmax = 6
        self.nparams = nparams = 7
        
        self.acteddy = acteddy = {}
        self.neles = neles = {}
        
        self.vorts = []
        eles = []
        
        self.vortrad = vortrad = 0.04
        
        # add a single vortex
        self.vorts.append({'xinit': 0.3, 'yinit': 0.5, 'zinit': 0.5, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.7, 'zinit': 0.4, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.1, 'zinit': 0.6, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.2, 'zinit': 0.1, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.3, 'zinit': 0.9, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        self.vorts.append({'xinit': 0.3, 'yinit': 0.2, 'zinit': 0.3, 'tinit': 0.0, 'eps': 0.01, 'eles': {}})
        
        eset = {}
    
        # add elements to vortices
        for vort in self.vorts:
            for etype, eles in intg.system.ele_map.items():
                vort['eles'][etype] = set()
                box = BoxRegion([vort['xinit']-vortrad,vort['yinit']-vortrad,vort['zinit']-vortrad],[self.xmax,vort['yinit']+vortrad,vort['zinit']+vortrad])
                pts = eles.ploc_at_np('upts')
                pts = np.moveaxis(pts, 1, -1) # required = check with Freddie
                inside = box.pts_in_region(pts)
                elestemp = []
                if np.any(inside):
                    elestemp = np.any(inside, axis=0).nonzero()[0].tolist()
                for eid in elestemp:
                    xmin = pts[:,eid,0].min()
                    xmax = pts[:,eid,0].max()
                    ts = max(vort['tinit'], vort['tinit'] + ((xmin - vort['xinit'] - self.vortrad)/self.xvel))
                    te = ts + (xmax-xmin+2*self.vortrad)/self.xvel
                    vort['eles'][etype].add((eid, ts, te)) # tuple (eid,ts,te)
        
        # add macro and external data
        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence', {'nvmax': nvmax, 'vortrad': vortrad, 'xvel': xvel})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                               
        # keep info for later (dont actually need to do this, can just look at matrix dims)                      
        for etype, eles in intg.system.ele_map.items():
            self.neles[etype] = eles.neles
                                 
    def __call__(self, intg):
        tcurr = intg.tcurr
        
        if tcurr >= self.tnext:
            print('Transferring update ...')
            for etype, neles in self.neles.items():

        	    temp = np.zeros((self.nvmax, self.nparams, neles))
        	    ctemp = np.zeros(neles).astype(int) # keep track of which vortex entry we are at for each element

        	    for vort in self.vorts:
        	        elestemp = vort['eles'][etype]
        	        for ele in elestemp:
        	            eid = ele[0]
        	            ts = ele[1]
        	            te = ele[2]
        	            if ts < (self.tnext+self.twin) and te > self.tnext:
        	                if ctemp[eid] >= self.nvmax:
        	                    print("Error ...")
        	                temp[ctemp[eid],0,eid]=vort['xinit']
        	                temp[ctemp[eid],1,eid]=vort['yinit']
        	                temp[ctemp[eid],2,eid]=vort['zinit']
        	                temp[ctemp[eid],3,eid]=vort['tinit']
        	                temp[ctemp[eid],4,eid]=vort['eps']  
        	                temp[ctemp[eid],5,eid]=ts
        	                temp[ctemp[eid],6,eid]=te
        	                ctemp[eid] += 1
        	            else:
        	                print('rejecting')
        	                
        	    self.acteddy[etype].set(temp)
        
        self.tnext += self.twin
    	    
