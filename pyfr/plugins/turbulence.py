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

        comm, rank, root = get_comm_rank_root()
        #print(rank)
        
        self.restart = restart
        self.tnext = 0.0
        self.tfull = {}
        self.tend = 1.0

        self.drtol = 0.1

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

        print(f'nvorts {self.nvorts}')
        
        self.dtmargin = 0
        
        if hasattr(intg, 'dtmax'):
            self.dtmargin = intg.dtmax
        else:
            self.dtmargin = intg._dt

        #self.nvmax = nvmax = 20
        self.nparams = nparams = 9
        
        self.neles = {}
        self.eles = {}
        self.pts = {}
        self.etypeupdate = {}
        self.temp = {}
  
        bbox = BoxRegion([self.xmin,self.ymin,self.zmin],[self.xmax,self.ymax,self.zmax])
        
        for etype, eles in intg.system.ele_map.items():
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1)
            inside = bbox.pts_in_region(pts)
            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                #print(eids)
                #print(etype)
                self.eles[etype] = eids
                self.pts[etype] = pts[:,eids,:]
                self.tfull[etype] = 0.0
                self.neles[etype] = eles.neles
       
        if not bool(self.eles):
           self.tnext = math.inf

        self.rng = np.random.default_rng(42)
        
        ###############
        self.vorts = []
        ###############
        
        vid = 0
        while vid <= self.nvorts:
            #print(f'making vort history {vid}')
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
            vbox = BoxRegion([self.xmin,
                                 vort[1]-self.ls,
                                 vort[2]-self.ls],
                                [self.xmax,
                                 vort[1]+self.ls,
                                 vort[2]+self.ls])
            #print(f'Adding vort {vid} to lut')
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

        self.nvmaxauto = self.dryrun()
        print(self.nvmaxauto)

        ###########################
        self.acteddy = acteddy = {}
        ###########################
        
        for etype in self.eles:
            comm, rank, root = get_comm_rank_root()
            print(f'adding kernels for rank {rank}')
            eles = intg.system.ele_map[etype]
            self.temp[etype] = np.full((self.nvmaxauto[etype], self.nparams, eles.neles), 0.0)
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
            {'nvmax': self.nvmaxauto[etype], 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
             'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
             'sigma' : sigma, 'rs': rs, 'gc': gc})
            acteddy[etype] = eles._be.matrix((self.nvmaxauto[etype], nparams, self.neles[etype]), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{self.nvmaxauto[etype]}][{nparams}]',
                               value=acteddy[etype])

    def dryrun(self):
        nvmaxmin = {}
        comm, rank, root = get_comm_rank_root()
        for etype, luts in self.lut.items():
            nvmaxmin[etype] = 0
            for leid, lut in luts.items():
                for i, act in enumerate(lut):
                    ifidx = next((j for j,x in enumerate(lut) if x[1] > act[2]+self.drtol),0)
                    if ifidx:
                        tot = ifidx - i + 1
                    else:
                        tot = len(lut) - i
                    if tot > nvmaxmin[etype]:
                        #print(tot)
                        nvmaxmin[etype] = tot
            #print(f'nvmaxmin for {etype} on rank {rank} is {nvmaxmin[etype]}')
        return nvmaxmin
                          
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        
        if tcurr+self.dtmargin >= self.tnext:
            for etype, luts in self.lut.items():
                if self.tfull[etype] <= self.tnext:
                    maxadv = np.inf
                    for leid, lut in luts.items():
                        if lut:
                            geid = self.eles[etype][leid]
                            idx = next((i for i,x in enumerate(self.temp[etype][:,8,geid]) if x > tcurr),len(self.temp[etype][:,8,geid]))
                            if idx:
                                self.temp[etype][:,:,geid] = np.roll(self.temp[etype][:,:,geid], -idx, axis=0)
                               
                                front = np.array(lut[:idx])

                                fronti = front[:,0].astype(int)
                            
                                del self.lut[etype][leid][:idx]
                                
                                vortnp = np.array(self.vorts)
                                
                                selvorts = vortnp[fronti,:]
                                
                                add = np.concatenate((selvorts, front[:,-2:]), axis=1)
                                add = np.pad(add, [(0,idx-add.shape[0]),(0,0)], 'constant')

                                self.temp[etype][-idx:,:,geid] = add

                                # if there is something left to add from lut check when we need to come
                                if self.lut[etype][leid] and (self.temp[etype][-1,7,geid] < maxadv):
                                    
                                    maxadv = self.temp[etype][-1,7,geid]

                    if maxadv <= self.tnext:
                        print('ERROR, nvmax too small')

                    self.tfull[etype] = maxadv

                    self.acteddy[etype].set(self.temp[etype])
                    comm, rank, root = get_comm_rank_root()
                    print(f'rank {rank}, etype {etype}, maxadv {maxadv}')
                    #print(maxadv)
                    
            self.tnext = min(val for val in self.tfull.values())
            print(f'rank {rank}, tnext {self.tnext}')
