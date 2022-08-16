# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import uuid

from collections import OrderedDict
from pyfr.plugins.base import BasePlugin
from pyfr.regions import BoxRegion
from pyfr.mpiutil import get_comm_rank_root

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['navier-stokes']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix, restart=False):
        super().__init__(intg, cfgsect, suffix)
        
        self.restart = restart
        self.twindowmax = 10.0
        
        self.comm, self.rank, self.root = get_comm_rank_root()
        constants = self.cfg.items_as('constants', float)
        params = self.cfg.items_as(cfgsect, float)

        self.rhobar = rhobar = params['rho-bar']
        self.ubar = ubar = params['u-bar']
        self.machbar = machbar = params['mach-bar']
        self.rs = rs = params['reynolds-stress']
        self.ls = ls = params['length-scale']
        self.sigma = sigma = params['sigma']
        
        self.xin = xin = params['x-in']
        self.xmin = xmin = xin - ls
        self.xmax = xmax = xin + ls
        self.ymin = ymin = params['y-min']
        self.ymax = ymax = params['y-max']
        self.zmin = zmin = params['z-min']
        self.zmax = zmax = params['z-max']

        self.srafac = srafac = rhobar*(constants['gamma']-1.0)*machbar*machbar
        self.gc = gc = (1.0/(4.0*math.sqrt(math.pi)*sigma))*math.erf(1.0/sigma)
        self.nvorts = nvorts = int((ymax-ymin)*(zmax-zmin)/(4*self.ls*self.ls))
        
        self.dtmargin = 0
        
        if hasattr(intg, 'dtmax'):
            self.dtmargin = intg.dtmax
        else:
            self.dtmargin = intg._dt

        self.nvmax = nvmax = 20
        self.nparams = nparams = 9

        self.mesh = intg.system.ele_map.items()
        
        self.neles = {}
        
        for etype, eles in self.mesh:
            self.neles[etype] = eles.neles


        self.tnext = 0.0
        self.tfull = []
        self.etypeupdate = {}
        for etype in intg.system.ele_map:
            self.etypeupdate[etype] = True

        # new code
        
        self.tend = 4.0
        
        self.rng = np.random.default_rng(42)
        self.tadv = {}
        
        ###############
        self.vorts = []
        ###############
        
        vid = 0
        while vid <= self.nvorts:
            t = 0.0
            initial = True
            while t < self.tend:
                if initial:
                    xinit = self.xmin + (self.xmax-self.xmin)*self.rng.random()
                else:
                    xinit = self.xmin
                yinit = self.ymin + (self.ymax-self.ymin)*self.rng.random()
                zinit = self.zmin + (self.zmax-self.zmin)*self.rng.random()
                epsx = self.rng.choice([-1,1])
                epsy = self.rng.choice([-1,1])
                epsz = self.rng.choice([-1,1])
                self.vorts.append([xinit,yinit,zinit,t,epsx,epsy,epsz])
                t += (self.xmax-xinit)/self.ubar
                initial = False
            vid += 1
        
        #############
        self.lut = {}
        #############
        
        for etype in intg.system.ele_map:
            self.lut[etype] = {}
            
        for vid, vort in enumerate(self.vorts):
            for etype, eles in self.mesh:
                elestemp = []
                box = BoxRegion([self.xmin,
                                 vort[1]-self.ls,
                                 vort[2]-self.ls],
                                [self.xmax,
                                 vort[1]+self.ls,
                                 vort[2]+self.ls])                
                pts = eles.ploc_at_np('upts')
                pts = np.moveaxis(pts, 1, -1)
                inside = box.pts_in_region(pts)

                if np.any(inside):
                    elestemp = np.any(inside, axis=0).nonzero()[0].tolist()
                    
                for eid in elestemp:
                    exmin = pts[:,eid,0].min()
                    exmax = pts[:,eid,0].max()
                    ts = max(vort[3], vort[3] + ((exmin - vort[0] - self.ls)/self.ubar))
                    te = ts + (exmax-exmin+2*self.ls)/self.ubar
                    if eid not in self.lut[etype]:
                        self.lut[etype][eid] = []   
                    self.lut[etype][eid].append([vid,ts,te])
        
        for etype, eles in self.mesh:
           for k, v in self.lut[etype].items():
               v.sort(key=lambda x: x[1],reverse=True)

        ###########################
        self.acteddy = acteddy = {}
        ###########################
        
        for etype, eles in self.mesh:
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
            {'nvmax': nvmax, 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
             'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
             'sigma' : sigma, 'rs': rs, 'gc': gc})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                          
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        
        if tcurr+self.dtmargin >= self.tnext:
            # cull front of lut
            for etype, eles in self.mesh:
                for eid in self.lut[etype]:
                    while self.lut[etype][eid] and (self.lut[etype][eid][-1][2] < tcurr):
                        self.lut[etype][eid].pop()
                           
            for etype, neles in self.neles.items():      
                if self.etypeupdate[etype]:
                    temp = np.zeros((self.nvmax, self.nparams, neles))
                    for eid in self.lut[etype]:
                        lutl = len(self.lut[etype][eid])
                        lutlm = min(lutl,self.nvmax)
                        for i in range(lutlm):
                            temp[i,:,eid] = self.vorts[self.lut[etype][eid][lutl-i][0]] + self.lut[etype][eid][lutl-i][-2:]
                    adv = np.min(temp[self.nvmax-1,7,:][np.nonzero(temp[self.nvmax-1,7,:])])
                    # need to hande case where we have run everything towards the end of the run
                    if adv <= self.tnext:
                        print('Increase nvmax')
                    self.tfull.append({'etype': etype, 't': adv})
                    self.acteddy[etype].set(temp)
                    self.etypeupdate[etype] = False
                    
            self.tfull.sort(key=lambda x: x["t"])        
            tf = self.tfull.pop(0)
            self.tnext = tf['t']
            self.etypeupdate[tf['etype']] = True
            while self.tfull:
                if self.tfull[0]['t'] == tf:
                    tf = self.tfull.pop(0)
                    self.etypeupdate[tf['etype']] = True
                else:
                    break
                    
