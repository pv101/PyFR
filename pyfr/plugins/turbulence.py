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
        self.tnxt = 0.0
        self.trcl = {}
        self.tend = 1.0

        self.btol = 0.1

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

        self.dtol = 0
        
        if hasattr(intg, 'dtmax'):
            self.dtol = intg.dtmax
        else:
            self.dtol = intg._dt

        self.nparams = nparams = 9
        
        self.neles = {}
        self.geid = {}
        self.pts = {}
        self.etypeupdate = {}
        self.buff = {}
  
        bbox = BoxRegion([self.xmin,self.ymin,self.zmin],[self.xmax,self.ymax,self.zmax])
        
        for etype, eles in intg.system.ele_map.items():
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1)
            inside = bbox.pts_in_region(pts)
            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                self.geid[etype] = eids
                self.pts[etype] = pts[:,eids,:]
                self.trcl[etype] = 0.0
                self.neles[etype] = eles.neles
       
        if not bool(self.geid):
           self.tnxt = math.inf

        self.rng = np.random.default_rng(42)
        
        #################
        # Make vortices #
        #################
        
        vid = 0
        temp = []
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
                temp.append([xinit,yinit,zinit,t,epsx,epsy,epsz])
                t += (self.xmax-xinit)/self.ubar
                initial = False
            vid += 1

        self.vdat = np.asarray(temp)

        #################
        # Make acts #
        #################
        
        ttlut = defaultdict(lambda: defaultdict(list))
            
        for vid, vort in enumerate(self.vdat):
            vbox = BoxRegion([self.xmin,
                                 vort[1]-self.ls,
                                 vort[2]-self.ls],
                                [self.xmax,
                                 vort[1]+self.ls,
                                 vort[2]+self.ls])
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
                    ttlut[etype][eid].append([vid,ts,te])
               
        for luts in ttlut.values():
           for v in luts.values():
               v.sort(key=lambda x: x[1])

        self.acts = defaultdict(lambda: defaultdict(dict))

        for k,v in ttlut.items():
            for kk, vv in v.items():
                nvv = np.array(vv) 
                self.acts[k][kk]['vidx'] = nvv[:,0].astype(int)
                self.acts[k][kk]['vtim'] = nvv[:,-2:] 

        self.nvmx = self.getnvmx()

        self.acteddy = acteddy = {}
        
        for etype in self.geid:
            eles = intg.system.ele_map[etype]
            self.buff[etype] = np.full((self.nvmx[etype], self.nparams, eles.neles), 0.0)
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
            {'nvmax': self.nvmx[etype], 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
             'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
             'sigma' : sigma, 'rs': rs, 'gc': gc})
            acteddy[etype] = eles._be.matrix((self.nvmx[etype], nparams, self.neles[etype]), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{self.nvmx[etype]}][{nparams}]',
                               value=acteddy[etype])

    def getnvmx(self):
        nvmx = {}
        for etype, acts in self.acts.items():
            nvmx[etype] = 0
            for leid, actl in acts.items():
                for i, te in enumerate(actl['vtim'][:,1]):
                    shft = next((j for j,v in enumerate(actl['vtim'][:,0]) if v > te+self.btol),len(actl)-1) - i + 1
                    if shft > nvmx[etype]:
                        nvmx[etype] = shft
        return nvmx
                          
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        if tcurr+self.dtol >= self.tnxt:
            for etype, acts in self.acts.items():
                if self.trcl[etype] <= self.tnxt:
                    trcl = np.inf
                    for leid, actl in acts.items():
                        if actl['vidx'].any():
                            geid = self.geid[etype][leid]
                            shft = next((i for i,v in enumerate(self.buff[etype][:,8,geid]) if v > tcurr),self.nvmx[etype])
                            if shft:
                                newb = np.concatenate((self.vdat[actl['vidx'][:shft],:], actl['vtim'][:shft,:]), axis=1)
                                newb = np.pad(newb, [(0,shft-newb.shape[0]),(0,0)], 'constant')
                                self.buff[etype][:,:,geid] = np.concatenate((self.buff[etype][shft:,:,geid],newb))
                                actl['vidx'] = actl['vidx'][shft:]
                                actl['vtim'] = actl['vtim'][shft:,:]
                                if actl['vidx'].any() and (self.buff[etype][-1,7,geid] < trcl):
                                    trcl = self.buff[etype][-1,7,geid]
                    self.trcl[etype] = trcl
                    self.acteddy[etype].set(self.buff[etype])
            self.tnxt = min(v for v in self.trcl.values())
