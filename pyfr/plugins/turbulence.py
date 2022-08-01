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

        self.vortchains = []
        self.buff = {}
        self.tfull = []
        self.etypeupdate = {}
        for etype in intg.system.ele_map:
            self.buff[etype] = []
            self.etypeupdate[etype] = True
        
        self.mesh = intg.system.ele_map.items()
        
        self.neles = {}
        
        for etype, eles in intg.system.ele_map.items():
            self.neles[etype] = eles.neles

        self.tnext = 0.0
        
        if self.restart:
            pass
        else:
            for i in range(self.nvorts):
                self.make_vortchain(0.0, i)

        # add macro and external data

        self.acteddy = acteddy = {}
        
        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
            {'nvmax': nvmax, 'ls': ls, 'ubar': ubar, 'srafac': srafac, 'xin': xin,
             'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
             'sigma' : sigma, 'rs': rs})
            acteddy[etype] = eles._be.matrix((nvmax, nparams, eles.neles), tags={'align'})
            eles._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{nparams}]',
                               value=acteddy[etype])
                               
    def make_vort(self, initial=True)
        vid = uuid.uuid1()
        xyz = np.random.uniform(0, 1, 3)
        if initial:
            xinit = self.xmin + (self.xmax-self.xmin)*xyz[0]
        else:
            xinit = self.xmin
        yinit = self.ymin + (self.ymax-self.ymin)*xyz[1]
        zinit = self.zmin + (self.zmax-self.zmin)*xyz[2]
        tdead = t + (self.xmax-xinit)/self.ubar
        epsx = np.random.choice([-1,1])
        epsy = np.random.choice([-1,1])
        epsz = np.random.choice([-1,1])
        vort = {'vcid': vcid, 'vid': vid, 'xinit': xinit,
                'yinit': yinit, 'zinit': zinit, 'tinit': t,
                'tdead': tdead, 'epsx': epsx, 'epsy': epsy,
                'epsz': epsz}
        return vort

    def make_vortchain(self, tinit, vcid, clen=2):
        t = tinit
        vortchain = OrderedDict()
        seed = vcid
        np.random.seed(seed)
        for i in range(clen):
            vort = make_vort()
            vortchain[vid] = vort
            self.vort_to_buffer(vort)
            t += tdead
        self.vortchains.append(vortchain)
        
    def load_vortchain(self, vortchains):
        self.vortchains = vortchains
        for vortchain in self.vortchains:
            for vort in vortchain:
                self.vort_to_buffer(vort)
           
    def vort_to_vortchain(self, vcid):
        t = self.vortchains[vcid][next(reversed(self.vortchains[vcid]))]['tdead']
        seed = int(f'{t:.6E}'.split('E')[0].replace('.', ''))
        np.random.seed(seed)
        vort = make_vort(False)
        self.vort_to_buffer(vort)
        self.vortchains[vcid][vid]=vort

    def vort_to_buffer(self, vort):
        for etype, eles in self.mesh:
            elestemp = []
            box = BoxRegion([vort['xinit']-self.ls,
                             vort['yinit']-self.ls,
                             vort['zinit']-self.ls],
                            [vort['xinit']+self.ls,
                             vort['yinit']+self.ls,
                             vort['zinit']+self.ls])                
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1)
            inside = box.pts_in_region(pts)

            if np.any(inside):
                elestemp = np.any(inside, axis=0).nonzero()[0].tolist()
                
            for eid in elestemp:
                exmin = pts[:,eid,0].min()
                exmax = pts[:,eid,0].max()
                ts = max(vort['tinit'], vort['tinit'] + ((exmin - vort['xinit'] - self.ls)/self.ubar))
                te = min(vort['tdead'], ts + (exmax-exmin+2*self.ls)/self.ubar)
                self.buff[etype].append({'action': 'push', 'vcid': vort['vcid'],
                                         'vid': vort['vid'], 'eid': eid, 'ts': ts,
                                         'te': te, 't': ts}) 
            
            self.buff[etype].append({'action': 'dead', 'vcid': vort['vcid'], 'te': vort['tdead'],
                                     't': vort['tdead']})
                                     
            self.buff[etype].sort(key=lambda x: x["t"])
                               
    def __call__(self, intg):
        
        tcurr = intg.tcurr

        if tcurr+self.dtmargin >= self.tnext:
            # cull vorts from vortchains
            for vortchain in self.vortchains:
                while list(vortchain.values())[0]['tdead'] < self.tnext:
                    vortchain.popitem(last=False)
            for etype, neles in self.neles.items():
                # check if etype needs updating        
                if self.etypeupdate[etype]:
                    print(f'Updating buffer for {etype} on rank {self.rank} for time {self.tnext}')
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
            	            vort = self.vortchains[vcid][vid]
            	            temp[ctemp[eid],0,eid]=vort['xinit']
            	            temp[ctemp[eid],1,eid]=vort['yinit']
            	            temp[ctemp[eid],2,eid]=vort['zinit']
            	            temp[ctemp[eid],3,eid]=vort['tinit']
            	            temp[ctemp[eid],5,eid]=act['ts']
            	            temp[ctemp[eid],6,eid]=act['te']
            	            temp[ctemp[eid],4,eid]=vort['epsx']
            	            temp[ctemp[eid],7,eid]=vort['epsy']
            	            temp[ctemp[eid],8,eid]=vort['epsz']
            	            ctemp[eid] += 1
            	            if ctemp[eid] == self.nvmax:
            	                if act['ts'] <= self.tnext:
            	                    print('Increase nvmax')
            	                # record ts associated with maxed out temp for given etype
            	                self.tfull.append({'etype': etype, 't': act['ts']})
            	                self.tfull.sort(key=lambda x: x["t"])
            	                self.buff[etype] = list(filter(lambda x: x['te'] >= act['ts'], self.buff[etype]))
            	                
            	                # break the while loop for that particular element type
            	                break
                        elif act['action'] == 'dead':
        	                vcid = act['vcid']
        	                # add a new vortx to the chain so it doesnt risk running dry
        	                # note that it will add events to the buffer, but these will all
        	                # be in the future, so buffloc is still valid
        	                self.vort_to_vortchain(vcid)
        	                
                    self.acteddy[etype].set(temp)
                    self.etypeupdate[etype] = False
                    
            # make flags
            ac = self.tfull.pop(0)
            self.tnext = ac['t']
            self.etypeupdate[ac['etype']] = True
            while self.tfull:
                if self.tfull[0]['t'] == ac:
                    ac = self.tfull.pop(0)
                    self.etypeupdate[ac['etype']] = True
                else:
                    break

