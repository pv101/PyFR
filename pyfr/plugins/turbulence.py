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
        
        self.tnxt = intg.tcurr
        self.trcl = {}
        self.tend = intg.tend
        self.tstart = intg.tstart

        print(self.tstart)
        print(self.tnxt)
        print(self.tend)

        self.btol = 0.1

        constants = self.cfg.items_as('constants', float)
        params = self.cfg.items_as(cfgsect, float)

        self.rhobar = rhobar = params['rho-bar']
        self.ubar = ubar = params['u-bar']
        self.machbar = machbar = params['mach-bar']
        self.rootrs = rootrs = np.sqrt(params['reynolds-stress'])
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
        #gcold = (1.0/(4.0*math.sqrt(math.pi)*sigma))*math.erf(1.0/sigma)
        self.gc = gc = math.sqrt((2.0*sigma/(math.sqrt(math.pi)))*(1.0/math.erf(1.0/sigma)))

        #print(gcold)
        #print(gc)

        (1.0/(4.0*math.sqrt(math.pi)*sigma))*math.erf(1.0/sigma)


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
        self.buff = {}
  
        bbox = BoxRegion([self.xmin,self.ymin,self.zmin],[self.xmax,self.ymax,self.zmax])
        
        self.seed = 42

        self.rng = np.random.default_rng(self.seed)
        

        self.tbegin = intg.tcurr

        #################
        # Make vortices #
        #################
        
        vid = 0
        temp = []
        while vid <= self.nvorts:
            t = self.tstart # start right at the start
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
                if t >= self.tbegin:
                    temp.append([xinit,yinit,zinit,t,epsx,epsy,epsz])
                t += (self.xmax-xinit)/self.ubar
                initial = False
            vid += 1

        self.vdat = np.asarray(temp)

        #####################
        # Make action buffer#
        #####################

        self.uberbuff = []

        for etype, eles in intg.system.ele_map.items():
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1)
            inside = bbox.pts_in_region(pts)

            ttlut = defaultdict(list)
            vidx = defaultdict()
            vtim = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                ptsr = pts[:,eids,:]
                for vid, vort in enumerate(self.vdat):
                    vbox = BoxRegion([self.xmin,
                                         vort[1]-self.ls,
                                         vort[2]-self.ls],
                                        [self.xmax,
                                         vort[1]+self.ls,
                                         vort[2]+self.ls])

                    
                    elestemp = []               
                    insidev = vbox.pts_in_region(ptsr)

                    if np.any(insidev):
                        elestemp = np.any(insidev, axis=0).nonzero()[0].tolist() # box local indexing
                        
                    for eid in elestemp:
                        exmin = ptsr[:,eid,0].min()
                        exmax = ptsr[:,eid,0].max()
                        ts = max(vort[3], vort[3] + ((exmin - vort[0] - self.ls)/self.ubar))
                        te = ts + (exmax-exmin+2*self.ls)/self.ubar
                        ttlut[eid].append([vid,ts,te])

                    for kk, vv in ttlut.items():
                        vv.sort(key=lambda x: x[1])
                        nvv = np.array(vv)
                        vidx[kk] = nvv[:,0].astype(int)
                        vtim[kk] = nvv[:,-2:]

                nvmx = 0
                for leid, actl in vtim.items():
                    for i, te in enumerate(actl[:,1]):
                        shft = next((j for j,v in enumerate(actl[:,0]) if v > te+self.btol),len(actl)-1) - i + 1
                        if shft > nvmx:
                            nvmx = shft

                buff = np.full((nvmx, self.nparams, eles.neles), 0.0)

                adduberbuff = {'geid': eids, 'pts': ptsr, 'trcl': 0.0, 'neles': eles.neles, 'etype': etype, 'vidx': vidx, 'vtim': vtim, 'nvmx': nvmx, 'buff': buff, 'acteddy': eles._be.matrix((nvmx, nparams, eles.neles), tags={'align'})}
       
                self.uberbuff.append(adduberbuff)

                eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
                {'nvmax': nvmx, 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
                 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
                 'sigma' : sigma, 'rootrs': rootrs, 'gc': gc})

                eles._set_external('acteddy',
                                   f'in broadcast-col fpdtype_t[{nvmx}][{nparams}]',
                                   value=adduberbuff['acteddy'])




        if not bool(self.uberbuff):
           self.tnxt = math.inf
                     
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        if tcurr+self.dtol >= self.tnxt:
            for tid, thing in enumerate(self.uberbuff):    
                if thing['trcl'] <= self.tnxt:
                    trcl = np.inf
                    for leid, vidxs in thing['vidx'].items():
                        if vidxs.any():
                            geid = thing['geid'][leid]
                            shft = next((i for i,v in enumerate(thing['buff'][:,8,geid]) if v > tcurr),thing['nvmx'])
                            if shft:
                                newb = np.concatenate((self.vdat[vidxs[:shft],:], thing['vtim'][leid][:shft,:]), axis=1)
                                newb = np.pad(newb, [(0,shft-newb.shape[0]),(0,0)], 'constant')
                                self.uberbuff[tid]['buff'][:,:,geid] = np.concatenate((thing['buff'][shft:,:,geid],newb))
                                self.uberbuff[tid]['vidx'][leid] = vidxs[shft:]
                                self.uberbuff[tid]['vtim'][leid] = thing['vtim'][leid][shft:,:]
                                if self.uberbuff[tid]['vidx'][leid].any() and (self.uberbuff[tid]['buff'][-1,7,geid] < trcl):
                                    trcl = self.uberbuff[tid]['buff'][-1,7,geid]
                    self.uberbuff[tid]['trcl'] = trcl
                    self.uberbuff[tid]['acteddy'].set(thing['buff'])
            self.tnxt = min(etype['trcl'] for etype in self.uberbuff)
