# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import time
import uuid
from rtree import index

from collections import defaultdict
from numpy.lib.recfunctions import structured_to_unstructured
from pyfr.plugins.base import BasePlugin
from pyfr.regions import BoxRegion, RotatedBoxRegion
from pyfr.mpiutil import get_comm_rank_root

class pcg32rxs_m_xs:
    def __init__(self, seed):
        self.state = np.uint32(seed)
        self.multiplier = np.uint32(747796405)
        self.mcgmultiplier = np.uint32(277803737)
        self.increment = np.uint32(2891336453)
        self.opbits = np.uint8(4)
        self.b22 = np.uint8(22)
        self.b32 = np.uint8(32)
    def rand(self):
        oldstate = self.state
        self.state = (oldstate * self.multiplier) + self.increment
        rshift = np.uint8(oldstate >> (self.b32 - self.opbits))
        oldstate ^= oldstate >> (self.opbits + rshift)
        oldstate *= self.mcgmultiplier
        oldstate ^= oldstate >> self.b22
        return oldstate
    def random(self):
        return np.ldexp(self.rand(),-32)
    def getstate(self):
        return self.state
    def randint(self, a, b):
        return a + self.rand() % (b - a)
    def choice(self, seq):
        return seq[self.randint(0, len(seq))]
        
class Turbulence(BasePlugin):
    name = 'turbulence'
    systems = ['navier-stokes']
    formulations = ['std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend
        
        comm, rank, root = get_comm_rank_root()

        fdptype = intg.backend.fpdtype

        self.vortdtype = np.dtype([('loci', fdptype, 2), ('tinit', fdptype), ('state', np.uint32)])
        self.sstreamdtype = np.dtype([('vid', '<i4'), ('ts', fdptype), ('te', fdptype)])
        self.buffdtype = np.dtype([('tinit', fdptype), ('state', np.uint32), ('ts', fdptype), ('te', fdptype)])
        
        btol = 0.01

        gamma = self.cfg.getfloat('constants', 'gamma')
        rhobar = self.cfg.getfloat(cfgsect,'rho-bar')
        ubar = self.cfg.getfloat(cfgsect,'u-bar')
        machbar = self.cfg.getfloat(cfgsect,'mach-bar')
        rootrs = np.sqrt(self.cfg.getfloat(cfgsect,'reynolds-stress'))
        sigma = self.cfg.getfloat(cfgsect,'sigma')
        ls = self.cfg.getfloat(cfgsect,'length-scale')

        srafac = rhobar*(gamma-1.0)*machbar*machbar
        gc = math.sqrt((2.0*sigma/(math.sqrt(math.pi)))*(1.0/math.erf(1.0/sigma)))

        ydim = self.cfg.getfloat(cfgsect,'y-dim')
        zdim = self.cfg.getfloat(cfgsect,'z-dim')
        
        xmin = - ls
        xmax = ls
        ymin = -ydim/2.0
        ymax = ydim/2.0
        zmin = -zdim/2.0
        zmax = zdim/2.0

        nvorts = int((ymax-ymin)*(zmax-zmin)/(4*ls*ls))
        bbox = BoxRegion([xmin-ls,ymin-ls,zmin-ls],
                         [xmax+ls,ymax+ls,zmax+ls])
        
        theta = -1.0*np.radians(self.cfg.getfloat(cfgsect,'rot-angle'))

        periodicdim = self.cfg.get(cfgsect, 'periodic-dim', 'none')

        c = self.cfg.getliteral(cfgsect, 'centre')
        e = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))

        shift = np.array(c)
        rot=(np.cos(theta)*np.identity(3))+(np.sin(theta)*(np.cross(e, np.identity(3) * -1)))+(1.0-np.cos(theta))*np.outer(e,e)

        self.dtol = 0
        
        if hasattr(intg, 'dtmax'):
            self.dtol = intg.dtmax
        else:
            self.dtol = intg._dt

        seed = self.cfg.getint(cfgsect, 'seed')
        pcg32rng = pcg32rxs_m_xs(seed)
        
        ######################
        # Make vortex buffer #
        ######################
        
        vid = 0
        temp = []
        xtemp = []
        
        tinits = []
        
        print('Making tinits for vortices')
        start = time.time()

        while vid < nvorts:
            tinits.append(self.tstart + (xmax-xmin)*pcg32rng.random()/ubar)
            vid += 1
            
        end = time.time()
        print(f'It took {end-start} s')
               
        print('Making vortices')
        start = time.time()
          
        while True:     
            for vid, tinit in enumerate(tinits):
                #print(vid)
                state = pcg32rng.getstate()
                yinit = ymin + (ymax-ymin)*pcg32rng.random()
                zinit = zmin + (zmax-zmin)*pcg32rng.random()
                eps = 1.0*pcg32rng.randint(0,8)
                if tinit+((xmax-xmin)/ubar) >= self.tbegin and tinit <= self.tend:
                    xtemp.append(((yinit,zinit),tinit,state))
                tinits[vid] += (xmax-xmin)/ubar
            if all(tinit > self.tend for tinit in tinits):
                break
        
        self.vortbuff = np.asarray(xtemp, self.vortdtype)

        end = time.time()
        print(f'It took {end-start} s')
        
        #####################
        # Make action buffer#
        #####################

        self.actbuffs = []

        for etype, eles in intg.system.ele_map.items():
        
            print('Rotating and extracting {etype} in vortex box')
            start = time.time()
            
            neles = eles.neles
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, 0)
            ptsr = (rot @ (pts.reshape(3, -1) - shift[:,None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)
            inside = bbox.pts_in_region(ptsr)
            
            end = time.time()
            print(f'It took {end-start} s')

            stream = defaultdict(list)
            stream2 = defaultdict(list)
            sstream = defaultdict()
            sstream2 = defaultdict()
            
            
            


            if np.any(inside):
            
                print('Some {etype} in vortex box - getting eids and pts')
                start = time.time()
                
                eids = np.any(inside, axis=0).nonzero()[0] # eles in injection box
                ptsri = ptsr[:,eids,:] # points in injection box
                
                end = time.time()
                print(f'It took {end-start} s')
                
                
                print('New index stuff')
                start = time.time()
                
                p = index.Property()
                p.dimension = 3
                p.interleaved = True
                idx3d = index.Index(properties=p)
                
                nlspts = ptsri.shape[0]
                nleles = ptsri.shape[1]

                for i in range(nleles):
                    idx3d.insert(i,(ptsri[:,i,0].min(),ptsri[:,i,1].min(),ptsri[:,i,2].min(), ptsri[:,i,0].max(),ptsri[:,i,1].max(),ptsri[:,i,2].max()))
                
              
                
                end = time.time()
                print(f'It took {end-start} s')
                

                print('Checking hits for each vortex in injection box')
                start = time.time()
                
                for vid, vort in enumerate(self.vortbuff):
                
                    #print(f'Checking vortex {vid} of')

                    vbox = BoxRegion([xmin-ls, vort['loci'][0]-ls, vort['loci'][1]-ls],
                                     [xmax+ls, vort['loci'][0]+ls, vort['loci'][1]+ls])

                    elestemp = []
                    elestemp2 = []

                    #print('old search')
                    #start = time.time()
                    
                    #vinside = vbox.pts_in_region(ptsri)
                    #if np.any(vinside):
                    #    elestemp = np.any(vinside, axis=0).nonzero()[0].tolist()
                        
                    #end = time.time()
                    #print(f'It took {end-start} s')
                   
                   
                   
                   
                   
                   
                   
                    #print('searchtree') 
                    #start = time.time()
                       
                    rough = np.array(list(set(idx3d.intersection((xmin-ls, vort['loci'][0]-ls, vort['loci'][1]-ls, xmax+ls, vort['loci'][0]+ls, vort['loci'][1]+ls)))))   

                    if np.any(rough):
                        vinside2 = vbox.pts_in_region(ptsri[:,rough,:])
                        if np.any(vinside2):
                            elestemp2 = np.any(vinside2, axis=0).nonzero()[0].tolist()
                        
                    end = time.time()
                    #print(f'It took {end-start} s')
                    

                    
                    #for leid in elestemp:
                    #    exmin = ptsri[vinside[:,leid],leid,0].min()
                    #    exmax = ptsri[vinside[:,leid],leid,0].max()
                    #    ts = max(vort['tinit'], vort['tinit'] + ((exmin - xmin - ls)/ubar))
                     #   te = max(ts,min(ts + (exmax-exmin+2*ls)/ubar,vort['tinit']+((xmax-xmin)/ubar)))
                    #    stream[eids[leid]].append((vid,ts,te))
                        
                    for leid in elestemp2:
                        exmin = ptsri[vinside2[:,leid],rough[leid],0].min()
                        exmax = ptsri[vinside2[:,leid],rough[leid],0].max()
                        ts = max(vort['tinit'], vort['tinit'] + ((exmin - xmin - ls)/ubar))
                        te = max(ts,min(ts + (exmax-exmin+2*ls)/ubar,vort['tinit']+((xmax-xmin)/ubar)))
                        stream2[eids[rough[leid]]].append((vid,ts,te))
                        
        
                end = time.time()

                print(f'It took {end-start} s')
                
                
                
                

                print('Sorting vortices by tstart')
                start = time.time()
                
                #for k, v in stream.items():
                    #print(k)
                    #print(v)
                #    v.sort(key=lambda x: x[1]) 
                #    sstream[k] = np.asarray(v, self.sstreamdtype)
                    
                for k, v in stream2.items():
                    #print(k)
                    #print(v)
                    v.sort(key=lambda x: x[1]) 
                    sstream[k] = np.asarray(v, self.sstreamdtype)
                    
                end = time.time()
                print(f'It took {end-start} s')

                #print(sstream[0])
               # print(sstream2[0])


                print('Calculating nvmx')
                start = time.time()
                
                nvmx = 0
                for leid, actl in sstream.items():
                    for i, ts in enumerate(actl['ts']):
                        cnt = 0
                        while i-cnt >= 0:
                            if actl['te'][i-cnt] < ts:
                                break
                            cnt += 1
                        if cnt > nvmx:
                            nvmx = cnt
                nvmx += 1
                
                end = time.time()
                print(f'It took {end-start} s')
                
                
                buff = np.zeros((nvmx, neles), self.buffdtype)

                actbuff = {'trcl': 0.0, 'sstream': sstream, 'nvmx': nvmx, 'buff': buff,
                           'tinit': eles._be.matrix((nvmx, 1, neles), tags={'align'}),
                           'state': eles._be.matrix((nvmx, 1, neles), tags={'align'}, dtype=np.uint32)}

                eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
                {'nvmax': nvmx, 'ls': ls, 'ubar': ubar, 'srafac': srafac,
                 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
                 'sigma' : sigma, 'rootrs': rootrs, 'gc': gc, 'rot': rot, 'shift': shift
                })

                eles._set_external('tinit',
                                   f'in broadcast-col fpdtype_t[{nvmx}][1]',
                                   value=actbuff['tinit'])
                                   
                eles._set_external('state',
                                   f'in broadcast-col uint32_t[{nvmx}][1]',
                                   value=actbuff['state'])

                self.actbuffs.append(actbuff)
                print(f'Rank = {rank}, etype = {etype}, nvorts = {nvorts}, nvmx = {nvmx}.')

        if not bool(self.actbuffs):
           self.tnext = math.inf
                  
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        if tcurr+self.dtol >= self.tnext:
            for abid, actbuff in enumerate(self.actbuffs):    
                if actbuff['trcl'] <= self.tnext:
                    trcl = np.inf
                    for geid, sstream in actbuff['sstream'].items():
                        if sstream['vid'].any():
                            tmp = actbuff['buff'][:,geid][actbuff['buff'][:,geid]['te'] > tcurr]        
                            shft = actbuff['nvmx']-len(tmp)   
                            if shft:
                                newb = np.zeros(shft, self.buffdtype)
                                temp = self.vortbuff[['tinit', 'state']][sstream['vid'][:shft]]
                                pad = shft-temp.shape[0]
                                newb[['tinit', 'state']] = np.pad(temp, (0,pad), 'constant')
                                newb[['ts', 'te']] = np.pad(sstream[['ts', 'te']][:shft], (0,pad), 'constant')
                                self.actbuffs[abid]['buff'][:,geid] = np.concatenate((tmp,newb))
                                self.actbuffs[abid]['sstream'][geid] = sstream[shft:]
                            else:
                                tstemp = sstream['ts'][0]
                                if tcurr >= tstemp:
                                    print(f'Active vortex in stream cannot fit on buffer.')
                                
                            if self.actbuffs[abid]['sstream'][geid]['vid'].any() and (self.actbuffs[abid]['buff'][-1,geid]['ts'] < trcl):
                                trcl = self.actbuffs[abid]['buff'][-1,geid]['ts']

                    self.actbuffs[abid]['trcl'] = trcl
                    self.actbuffs[abid]['tinit'].set(actbuff['buff']['tinit'][:, np.newaxis, :])
                    self.actbuffs[abid]['state'].set(actbuff['buff']['state'][:, np.newaxis, :])
            
            proptnext = min(etype['trcl'] for etype in self.actbuffs)
            if proptnext > self.tnext:
                self.tnext = proptnext
            else:
                print('Not advancing.')

