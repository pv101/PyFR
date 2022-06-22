# -*- coding: utf-8 -*-

import numpy as np
import uuid
import random

from collections import deque
from collections import OrderedDict
from pyfr.plugins.base import BasePlugin
from pyfr.regions import BoxRegion
from pyfr.mpiutil import get_comm_rank_root
from mpi4py import MPI

class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['*']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        self.comm, self.rank, self.root = get_comm_rank_root()
        
        self.buffloc = {}
        
        self.mesh = intg.system.ele_map.items()
        
        self.tnext = 0.0
        self.twin = 50.0
        
        self.restart = False
        
        self.xvel = xvel = 0.5
        
        self.nvorts = 15
        
        # the box
        self.xmin = 0.25
        self.xmax = 0.75
        self.ymin = 0.25
        self.ymax = 0.75
        self.zmin = 0.25
        self.zmax = 0.75
        
        self.nvmax = nvmax = 6
        self.nparams = nparams = 7
        
        self.acteddy = acteddy = {}
        self.neles = neles = {}
        
        self.vorts = []
        eles = []
        
        self.vortrad = vortrad = 0.04
        
        self.buff = {}
        
        self.n_ele_types = len(intg.system.ele_map)
        
        for etype in intg.system.ele_map:
            self.buff[etype] = []
        
        eset = {}
        
        if self.restart:
            pass
        else:
            for i in range(self.nvorts):
                self.make_vort_chain(intg, 0, i, 2)

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

    
    def make_vort_chain(self, intg, tinit, vcid, n):
        t = tinit
        vort_chain = OrderedDict()
        seed = vcid
        np.random.seed(seed)
        for i in range(n):
            xyz = np.random.uniform(0, 1, 3)

            #print(xyz)
            
            xinit = (self.xmin + self.vortrad) + (self.xmax-self.xmin-2*self.vortrad)*xyz[0]
            yinit = (self.ymin + self.vortrad) + (self.ymax-self.ymin-2*self.vortrad)*xyz[1]
            zinit = (self.zmin + self.vortrad) + (self.zmax-self.zmin-2*self.vortrad)*xyz[2]
            
            #print(f'rank={self.rank}, seed={seed}, x={xinit}, y={yinit}, z={zinit}')
            
            tdead = t + (self.xmax-xinit-self.vortrad)/self.xvel
            #print(tdead)
            
            vid = uuid.uuid1()
            vort = {'killcount': 0, 'vcid': vcid, 'vid': vid, 'xinit': xinit, 'yinit': yinit, 'zinit': zinit, 'tinit': t, 'tdead': tdead, 'eps': 0.01}
            vort_chain[vid] = vort
            self.vort_to_buffer(intg, vort)
            t += tdead
            
        self.vorts.append(vort_chain)
    
    def remove_vort_from_chain(self, intg, vcid, vid):
        #print(self.vorts[vcid])
        self.vorts[vcid][vid]['killcount'] += 1
        #print(self.vorts[vcid][vid]['killcount'])
        # if all the element types are done with it
        if self.vorts[vcid][vid]['killcount'] == self.n_ele_types:
            #print(f"Killing {vid} from chain {vcid}.")
            del self.vorts[vcid][vid] # delete
            #print(self.vorts[vcid])
            # then add one to end of the chain
            t = self.vorts[vcid][next(reversed(self.vorts[vcid]))]['tdead']
            seed = int(t*10000)
            
            #print(f'vcid={vcid}, seed={t}')
            np.random.seed(seed)
            xyz = np.random.uniform(0, 1, 3)
            
            xinit = self.xmin + self.vortrad
            yinit = (self.ymin + self.vortrad) + (self.ymax-self.ymin-2*self.vortrad)*xyz[1]
            zinit = (self.zmin + self.vortrad) + (self.zmax-self.zmin-2*self.vortrad)*xyz[2]
            
            #print(f'x={xinit}, y={yinit}, z={zinit}')
            
            tdead = t + (self.xmax-xinit-self.vortrad)/self.xvel
            
            vid = uuid.uuid1()
            
            vort = {'killcount': 0, 'vcid': vcid, 'vid': vid, 'xinit': xinit, 'yinit': yinit, 'zinit': zinit, 'tinit': t, 'tdead': tdead, 'eps': 0.01}
            self.vorts[vcid][vid]=vort
            self.vort_to_buffer(intg, vort)
            
    def add_vort_to_chain(self, intg, vcid):
        t = self.vorts[vcid][next(reversed(self.vorts[vcid]))]['tdead']
        seed = int(t*10000)
        
        np.random.seed(seed)
        xyz = np.random.uniform(0, 1, 3)
        
        xinit = self.xmin + self.vortrad
        yinit = (self.ymin + self.vortrad) + (self.ymax-self.ymin-2*self.vortrad)*xyz[1]
        zinit = (self.zmin + self.vortrad) + (self.zmax-self.zmin-2*self.vortrad)*xyz[2]
        
        #print(f'x={xinit}, y={yinit}, z={zinit}')
        
        tdead = t + (self.xmax-xinit-self.vortrad)/self.xvel
        
        vid = uuid.uuid1()
        
        vort = {'killcount': 0, 'vcid': vcid, 'vid': vid, 'xinit': xinit, 'yinit': yinit, 'zinit': zinit, 'tinit': t, 'tdead': tdead, 'eps': 0.01}
        self.vorts[vcid][vid]=vort
        self.vort_to_buffer(intg, vort)
       
    def vort_to_buffer(self, intg, vort):
        for etype, eles in self.mesh:

            elestemp = []

            box = BoxRegion([vort['xinit']-self.vortrad,
                             vort['yinit']-self.vortrad,
                             vort['zinit']-self.vortrad],
                            [self.xmax,
                             vort['yinit']+self.vortrad,
                             vort['zinit']+self.vortrad])
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1) # required = check with Freddie
            inside = box.pts_in_region(pts)

            if np.any(inside):
                elestemp = np.any(inside, axis=0).nonzero()[0].tolist()
                
            for eid in elestemp:
                xmin = pts[:,eid,0].min()
                xmax = pts[:,eid,0].max()
                ts = max(vort['tinit'], vort['tinit'] + ((xmin - vort['xinit'] - self.vortrad)/self.xvel))
                te = min(vort['tdead'], ts + (xmax-xmin+2*self.vortrad)/self.xvel)
                self.buff[etype].append({'action': 'push', 'vcid': vort['vcid'], 'vid': vort['vid'], 'eid': eid, 'ts': ts, 'te': te, 't': ts}) 
            
            self.buff[etype].append({'action': 'dead', 'vcid': vort['vcid'], 'vid': vort['vid'], 'ts': vort['tdead'], 'te': vort['tdead'], 't': vort['tdead']}) 
            self.buff[etype].sort(key=lambda x: x["t"])
                                    
    def __call__(self, intg):
        tcurr = intg.tcurr
        gmin = 0
        breaketype = None
        breakeid = None
        breaktime = []
        #print(f"Hello, {tcurr}. You are 1.")
        if tcurr >= self.tnext:
        
            for etype, neles in self.neles.items():
                temp = np.zeros((self.nvmax, self.nparams, neles))
                ctemp = np.zeros(neles).astype(int) # keep track of which vortex entry we are at for each element
                #print(etype)
                self.buffloc[etype] = 0
                while self.buff[etype]:
                    #act = self.buff[etype].pop(0)
                    act = self.buff[etype][self.buffloc[etype]]
                    self.buffloc[etype] += 1
                    if act['action'] == 'push':
        	            vcid = act['vcid']
        	            vid = act['vid']
        	            eid = act['eid']
        	            te = act['te']
        	            #print(f'VCID: {vcid}')
        	            print(f'vcid = {vcid}')
        	            print(f'te = {te}')
        	            vort = self.vorts[vcid][vid]
        	            temp[ctemp[eid],0,eid]=vort['xinit']
        	            temp[ctemp[eid],1,eid]=vort['yinit']
        	            temp[ctemp[eid],2,eid]=vort['zinit']
        	            temp[ctemp[eid],3,eid]=vort['tinit']
        	            temp[ctemp[eid],4,eid]=vort['eps']
        	            temp[ctemp[eid],5,eid]=act['ts']
        	            temp[ctemp[eid],6,eid]=act['te']
        	            ctemp[eid] += 1
        	            if ctemp[eid] == self.nvmax:
        	                #print(ctemp[eid])
        	                #print(etype)
        	                #print(eid)
        	                print(act['ts'])
        	                breaketype = etype
        	                breakeid = eid
        	                breaktime.append(act['ts']) #?
        	                break
                    elif act['action'] == 'dead':
    	                vcid = act['vcid']
    	                vid = act['vid']
    	                self.add_vort_to_chain(intg, vcid)
                self.acteddy[etype].set(temp)
                #print(f'rank={self.rank} and min = {min(breaktime)}')
                gmin = self.comm.allreduce(min(breaktime), op=MPI.MIN)
                #print(f'rank={self.rank} and globalmin = {gmin}')
                print(gmin)
                #ci = self.buff[etype].index(next(obj for obj in self.buff[etype] if obj['te'] >= gmin))
                self.buff[etype] = list(filter(lambda x: x['te'] >= gmin, self.buff[etype]))
                #self.buff[etype]=self.buff[etype][ci:]
                
                
                te0=self.buff[etype][0]['te']
                print(f'te0 = {te0}')

                # finally cull vorts from vortbuffer
                for vortchain in self.vorts:
                    while list(vortchain.values())[0]['tdead'] < gmin:
                        if(self.rank == 0):
                            tdead = list(vortchain.values())[0]['tdead']
                            vcid = list(vortchain.values())[0]['vcid']
                            print(f'front tdead of vcid {vcid} = {tdead} cf. {gmin}') 
                        popped = vortchain.popitem(last=False)
                        #pvid = popped['vid']
                        #pvcid = popped['vcid']
                        #print(f'popped {vid} from vc {vcid}')
                        if(self.rank == 0):
                            print(popped[0])
                            tdead = list(vortchain.values())[0]['tdead']
                            vcid = list(vortchain.values())[0]['vcid']
                            print(f'DONE front tdead of vcid {vcid} = {tdead} cf. {gmin}')

        	    # need to store the temp packed buffer, so we can cull old stuff but keep relevant stuff on next refill        
        
        
        
            print('Transferring update ...')
            #print(self.vorts)
            #print(self.buff)
            #for etype, neles in self.neles.items():
                #uuid = list(self.vorts[0].values())[0]['vid']
                #print(uuid)
                #self.remove_vort_from_chain(intg, 0, uuid)
          
            #for etype, neles in self.neles.items():

        	#    temp = np.zeros((self.nvmax, self.nparams, neles))
        	#    ctemp = np.zeros(neles).astype(int) # keep track of which vortex entry we are at for each element

        	#    for vort in self.vorts:
        	#        elestemp = vort['eles'][etype]
        	#        for ele in elestemp:
        	#            eid = ele[0]
        	#            ts = ele[1]
        	#            te = ele[2]
        	#            if ts < (self.tnext+self.twin) and te > self.tnext:
        	#                if ctemp[eid] >= self.nvmax:
        	#                    print("Error ...")
        	#                temp[ctemp[eid],0,eid]=vort['xinit']
        	#                temp[ctemp[eid],1,eid]=vort['yinit']
        	#                temp[ctemp[eid],2,eid]=vort['zinit']
        	#                temp[ctemp[eid],3,eid]=vort['tinit']
        	#                temp[ctemp[eid],4,eid]=vort['eps']  
        	#                temp[ctemp[eid],5,eid]=ts
        	#                temp[ctemp[eid],6,eid]=te
        	#                ctemp[eid] += 1
        	#            else:
        	#                print('rejecting')
        	                
        	#    self.acteddy[etype].set(temp)
        
            self.tnext = gmin
    	    
