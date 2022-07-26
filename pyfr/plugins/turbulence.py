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

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        self.restart = False
        
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

        self.nvmax = nvmax = 15
        self.nparams = nparams = 9
        
        self.buffloc = {}  
        
        self.mesh = intg.system.ele_map.items()
        
        self.acteddy = acteddy = {}
        self.neles = neles = {}
        
        self.vorts = []
        eles = []

        self.buff = {}
        
        self.n_ele_types = len(intg.system.ele_map)
        
        for etype in intg.system.ele_map:
            self.buff[etype] = []
        
        eset = {}
        
        self.tnext = 0.0
        
        if self.restart:
            pass
        else:
            for i in range(self.nvorts):
                self.make_vort_chain(intg, 0.0, i)

        # add macro and external data

        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
            {'nvmax': nvmax, 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
             'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
             'sigma' : sigma, 'rs': rs})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                               
        # keep info for later (dont actually need to do this, can just look at matrix dims)                      
        for etype, eles in intg.system.ele_map.items():
            self.neles[etype] = eles.neles
    
    def make_vort_chain(self, intg, tinit, vcid, n=2):
        t = tinit
        vort_chain = OrderedDict()
        seed = vcid
        np.random.seed(seed)
        for i in range(n):
            xyz = np.random.uniform(0, 1, 3)

            xinit = self.xmin + (self.xmax-self.xmin)*xyz[0]
            yinit = self.ymin + (self.ymax-self.ymin)*xyz[1]
            zinit = self.zmin + (self.zmax-self.zmin)*xyz[2]
            
            #print(f'rank={self.rank}, seed={seed}, x={xinit}, y={yinit}, z={zinit}')
            
            #tdead = t + (self.xmax-xinit-self.ls)/self.ubar
            tdead = t + (self.xmax-xinit)/self.ubar
            #print(tdead)
            
            vid = uuid.uuid1()
            epsx = float(np.random.choice([-1,1]))
            epsy = float(np.random.choice([-1,1]))
            epsz = float(np.random.choice([-1,1]))
            vort = {'vcid': vcid, 'vid': vid, 'xinit': xinit, 'yinit': yinit, 'zinit': zinit, 'tinit': t, 'tdead': tdead, 'eps': epsx, 'epsy': epsy, 'epsz': epsz}
            vort_chain[vid] = vort
            self.vort_to_buffer(intg, vort)
            t += tdead
            
        self.vorts.append(vort_chain)
           
    def add_vort_to_chain(self, intg, vcid):
        t = self.vorts[vcid][next(reversed(self.vorts[vcid]))]['tdead']
        seed = int(t*10000)
        
        np.random.seed(seed)
        xyz = np.random.uniform(0, 1, 3)
        
        xinit = self.xmin
        yinit = self.ymin + (self.ymax-self.ymin)*xyz[1]
        zinit = self.zmin + (self.zmax-self.zmin)*xyz[2]
        
        tdead = t + (self.xmax-xinit)/self.ubar
        
        vid = uuid.uuid1()
        
        epsx = float(np.random.choice([-1,1]))
        epsy = float(np.random.choice([-1,1]))
        epsz = float(np.random.choice([-1,1]))
        
        vort = {'vcid': vcid, 'vid': vid, 'xinit': xinit, 'yinit': yinit, 'zinit': zinit, 'tinit': t, 'tdead': tdead, 'eps': epsx, 'epsy': epsy, 'epsz': epsz}
        self.vorts[vcid][vid]=vort
        self.vort_to_buffer(intg, vort)
       
    def vort_to_buffer(self, intg, vort):
        for etype, eles in self.mesh:

            elestemp = []

            box = BoxRegion([vort['xinit']-self.ls,
                             vort['yinit']-self.ls,
                             vort['zinit']-self.ls],
                            [self.xmax,
                             vort['yinit']+self.ls,
                             vort['zinit']+self.ls])
            # need to take intersetction of this box with the injection region                 
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1) # required = check with Freddie
            inside = box.pts_in_region(pts)

            if np.any(inside):
                elestemp = np.any(inside, axis=0).nonzero()[0].tolist()
                
            for eid in elestemp:
                exmin = pts[:,eid,0].min()
                exmax = pts[:,eid,0].max()
                ts = max(vort['tinit'], vort['tinit'] + ((exmin - vort['xinit'] - self.ls)/self.ubar))
                te = min(vort['tdead'], ts + (exmax-exmin+2*self.ls)/self.ubar)
                self.buff[etype].append({'action': 'push', 'vcid': vort['vcid'], 'vid': vort['vid'], 'eid': eid, 'ts': ts, 'te': te, 't': ts}) 
            
            self.buff[etype].append({'action': 'dead', 'vcid': vort['vcid'],
                                     'vid': vort['vid'], 'ts': vort['tdead'],
                                     'te': vort['tdead'], 't': vort['tdead']}) 
            self.buff[etype].sort(key=lambda x: x["t"])
                                    
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        tfull = []
        tfullm = 0
        
        if tcurr+self.dtmargin >= self.tnext:
            
            for etype, neles in self.neles.items():
                temp = np.zeros((self.nvmax, self.nparams, neles))
                # keep track of which vortex entry we are at for each element
                ctemp = np.zeros(neles).astype(int)
                # position along buffer 
                buffloc = 0
                while self.buff[etype]:
                    act = self.buff[etype][buffloc]
                    buffloc += 1
                    if act['action'] == 'push':
        	            eid = act['eid']
        	            vcid = act['vcid']
        	            vid = act['vid']
        	            vort = self.vorts[vcid][vid]
        	            temp[ctemp[eid],0,eid]=vort['xinit']
        	            temp[ctemp[eid],1,eid]=vort['yinit']
        	            temp[ctemp[eid],2,eid]=vort['zinit']
        	            temp[ctemp[eid],3,eid]=vort['tinit']
        	            temp[ctemp[eid],4,eid]=vort['eps']
        	            temp[ctemp[eid],5,eid]=act['ts']
        	            temp[ctemp[eid],6,eid]=act['te']
        	            temp[ctemp[eid],7,eid]=vort['epsy']
        	            temp[ctemp[eid],8,eid]=vort['epsz']
        	            ctemp[eid] += 1
        	            if ctemp[eid] == self.nvmax:
        	                # record ts associated with maxed out bufer for given etype
        	                if act['ts'] == self.tnext:
        	                    print('Increaase nvmax')
        	                tfull.append(act['ts'])
        	                # break the while loop for that particular element type
        	                break
                    elif act['action'] == 'dead':
    	                vcid = act['vcid']
    	                # add a new vortx to the chain so it doesnt risk running dry
    	                self.add_vort_to_chain(intg, vcid)
                
                # send off the active eddy array
                self.acteddy[etype].set(temp)
                
            # agree on how far we can go
            
            #get limiting one
            tfullm = min(tfull, default=0)
                
            # cull old actions
            for etype, neles in self.neles.items():
                self.buff[etype] = list(filter(lambda x: x['te'] >= tfullm, self.buff[etype]))
            
            # cull old vorts
            for vortchain in self.vorts:
                while list(vortchain.values())[0]['tdead'] < tfullm:
                    vortchain.popitem(last=False)
                    
            print(f'Rank {self.rank}, tfullm  {tfullm}')          
            self.tnext = tfullm
    	    
