# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import uuid

from collections import defaultdict
from pyfr.plugins.base import BasePlugin
from pyfr.regions import BoxRegion, RotatedBoxRegion
from pyfr.mpiutil import get_comm_rank_root

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['navier-stokes']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix, restart=False):
        super().__init__(intg, cfgsect, suffix)
        
        self.restart = restart
        self.tnext = 0.0
        self.tfull = []
        self.tend = 4.0
        
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
        
        self.e = [0,0,1]
        self.c = [0.5,0.5,0.5]
        self.theta = 0
        
        theta = 1.0*np.radians(self.theta)
        
        qi = self.e[0]*np.sin(theta/2) 
        qj = self.e[1]*np.sin(theta/2)
        qk = self.e[2]*np.sin(theta/2)
        qr = np.cos(theta/2)
        
        a11 = 1.0 - 2.0*qj*qj - 2.0*qk*qk
        a12 = 2.0*(qi*qj - qk*qr)
        a13 = 2.0*(qi*qk + qj*qr)
        a21 = 2.0*(qi*qj + qk*qr)
        a22 = 1.0 - 2.0*qi*qi - 2.0*qk*qk
        a23 = 2.0*(qj*qk - qi*qr)
        a31 = 2.0*(qi*qk - qj*qr)
        a32 = 2.0*(qj*qk + qi*qr)
        a33 = 1.0 - 2.0*qi*qi - 2.0*qj*qj
        
        self.rot = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        
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
        
        self.neles = {}
        self.eles = {}
        self.pts = {}
        self.etypeupdate = {}
  
        bbox = RotatedBoxRegion([self.xmin,self.ymin,self.zmin],[self.xmax,self.ymax,self.zmax],self.e,self.c,self.theta)
        
        for etype, eles in intg.system.ele_map.items():
            self.etypeupdate[etype] = True
            self.neles[etype] = eles.neles
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1)
            inside = bbox.pts_in_region(pts)
            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                self.eles[etype] = eids
                self.pts[etype] = pts[:,eids,:]
       
        if not bool(self.eles):
           self.tnext = math.inf

        self.rng = np.random.default_rng(42)
        
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
        
        #################################################
        self.lut = defaultdict(lambda: defaultdict(list))
        #################################################
            
        for vid, vort in enumerate(self.vorts):
            vbox = RotatedBoxRegion([self.xmin,
                                 vort[1]-self.ls,
                                 vort[2]-self.ls],
                                [self.xmax,
                                 vort[1]+self.ls,
                                 vort[2]+self.ls], self.e, self.c, self.theta)
            for etype, pts in self.pts.items():
                elestemp = []               
                inside = vbox.pts_in_region(pts)

                if np.any(inside):
                    elestemp = np.any(inside, axis=0).nonzero()[0].tolist() # box local indexing
                    
                for eid in elestemp:
                    exmin = pts[:,eid,0].min()
                    exmax = pts[:,eid,0].max()
                    ts = max(vort[3], vort[3] + ((exmin - vort[0] - self.ls)/self.ubar))
                    te = ts + (exmax-exmin+2*self.ls)/self.ubar   
                    self.lut[etype][eid].append([vid,ts,te])
               
        for luts in self.lut.values():
           for v in luts.values():
               v.sort(key=lambda x: x[1])

        ###########################
        self.acteddy = acteddy = {}
        ###########################
        
        for etype in self.eles:
            eles = intg.system.ele_map[etype]
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
            {'nvmax': nvmax, 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
             'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
             'sigma' : sigma, 'rs': rs, 'gc': gc})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, self.neles[etype]), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                          
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        
        if tcurr+self.dtmargin >= self.tnext:            
            for etype, luts in self.lut.items():      
                if self.etypeupdate[etype]:
                    for lut in luts.values():
                        idx = next((i for i,x in enumerate(lut) if x[2] > tcurr),len(lut))
                        del lut[:idx]
                    temp = np.zeros((self.nvmax, self.nparams, self.neles[etype]))
                    for leid, lut in luts.items():
                        geid = self.eles[etype][leid]
                        for i, act in enumerate(lut):
                            #temp[i,:,geid] = (self.rot @ (self.vorts[act[0]][:3] - self.c) + self.c) + self.vorts[act[0]][-4:] + act[-2:]
                            temp[i,:,geid] = self.vorts[act[0]][:3] + self.vorts[act[0]][-4:] + act[-2:]
                            if i == self.nvmax-1:
                                break
                    tsmax = temp[self.nvmax-1,7,:]
                    if any(tsmax!=0):
                        adv = np.min(tsmax[tsmax!=0])
                    else:
                        adv = math.inf
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
                if self.tfull[0]['t'] <= tf+self.dtmargin:
                    tf = self.tfull.pop(0)
                    self.etypeupdate[tf['etype']] = True
                else:
                    break
                    
