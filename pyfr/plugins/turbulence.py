# -*- coding: utf-8 -*-

import math
import numpy as np
import random
import uuid

from collections import defaultdict
from numpy.lib.recfunctions import structured_to_unstructured
from pyfr.plugins.base import BasePlugin
from pyfr.regions import BoxRegion, RotatedBoxRegion
from pyfr.mpiutil import get_comm_rank_root

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

        fdptype = intg.backend.fpdtype

        self.vortdtype = np.dtype([('loci', fdptype, 3), ('ti', fdptype), ('eps', fdptype, 3)])
        self.xttlutdtype = np.dtype([('vid', '<i4'), ('ts', fdptype), ('te', fdptype)])
        self.buffdtype = np.dtype([('loci', fdptype, 3), ('ti', fdptype), ('eps', fdptype, 3), ('ts', fdptype), ('te', fdptype)])
        
        btol = 0.1
        nparams = 9

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
        rng = np.random.default_rng(seed)

        ######################
        # Make vortex buffer #
        ######################
        
        vid = 0
        temp = []
        xtemp = []

        while vid <= nvorts:
            t = self.tstart # start right at the start
            initial = True
            while t < self.tend:
                if initial:
                    xinit = xmin + (xmax-xmin)*rng.random()
                else:
                    xinit = xmin
                yinit = ymin + (ymax-ymin)*rng.random()
                zinit = zmin + (zmax-zmin)*rng.random()
                epsx = rng.choice([-1,1])
                epsy = rng.choice([-1,1])
                epsz = rng.choice([-1,1])
                if t >= self.tbegin:
                    xtemp.append(((xinit,yinit,zinit),t,(epsx,epsy,epsz)))
                    if periodicdim == 'y':
                        if yinit+ls>ymax:
                            xtemp.append(((xinit,ymin-(ymax-yinit),zinit),t,(epsx,epsy,epsz)))
                        if yinit-ls<ymin:
                            xtemp.append(((xinit,ymax+(yinit-ymin),zinit),t,(epsx,epsy,epsz)))
                    if periodicdim == 'z':
                        if zinit+ls>zmax:
                            xtemp.append(((xinit,yinit,zmin-(zmax-zinit)),t,(epsx,epsy,epsz)))
                        if zinit-ls<zmin:
                            xtemp.append(((xinit,yinit,zmax+(zinit-zmin)),t,(epsx,epsy,epsz)))
                t += (xmax-xinit)/ubar
                initial = False
            vid += 1

        self.xvortbuff = np.asarray(xtemp, self.vortdtype)

        #####################
        # Make action buffer#
        #####################

        self.actbuffs = []

        for etype, eles in intg.system.ele_map.items():

            neles = eles.neles
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, 0)
            ptsr = (rot @ (pts.reshape(3, -1) - shift[:,None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)
            inside = bbox.pts_in_region(ptsr)

            ttlut = defaultdict(list)
            xttlut = defaultdict()
            vidx = defaultdict()
            vtim = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0] # eles in injection box
                ptsri = ptsr[:,eids,:] # points in injection box
                for vid, vort in enumerate(self.xvortbuff):
                    vbox = BoxRegion([vort['loci'][0]-ls, vort['loci'][1]-ls, vort['loci'][2]-ls],
                                     [xmax+ls, vort['loci'][1]+ls, vort['loci'][2]+ls])
                    elestemp = []               
                    vinside = vbox.pts_in_region(ptsri)

                    if np.any(vinside):
                        elestemp = np.any(vinside, axis=0).nonzero()[0].tolist() # injection box local indexing
                        
                    for eid in elestemp:
                        exmin = ptsri[:,eid,0].min()
                        exmax = ptsri[:,eid,0].max()
                        ts = max(vort['ti'], vort['ti'] + ((exmin - vort['loci'][0] - ls)/ubar))
                        te = ts + (exmax-exmin+2*ls)/ubar
                        ttlut[eids[eid]].append((vid,ts,te))

                for k, v in ttlut.items():
                    v.sort(key=lambda x: x[1])
                    xttlut[k] = np.asarray(v, self.xttlutdtype)
                    nv = np.array(v)
                    vidx[k] = nv[:,0].astype(int)
                    vtim[k] = nv[:,-2:]

                # nvmx = 0
                # for leid, actl in vtim.items():
                #     for i, te in enumerate(actl[:,1]):
                #         shft = next((j for j,v in enumerate(actl[:,0]) if v > te+btol),len(actl)-1) - i + 1
                #         if shft > nvmx:
                #             nvmx = shft

                nvmx = 0
                for leid, actl in xttlut.items():
                    for i, te in enumerate(actl['te']):
                        shft = next((j for j,v in enumerate(actl['ts']) if v > te+btol),len(actl)-1) - i + 1
                        if shft > nvmx:
                            nvmx = shft

                #buff = np.full((nvmx, nparams, neles), 0.0)

                buff = np.zeros((nvmx, neles), self.buffdtype)


                #actbuff = {'trcl': 0.0, 'vidx': vidx, 'vtim': vtim, 'nvmx': nvmx, 'buff': buff, 'acteddy': eles._be.matrix((nvmx, nparams, neles), tags={'align'})}

                actbuff = {'trcl': 0.0, 'xttlut': xttlut, 'nvmx': nvmx, 'buff': buff, 'acteddy': eles._be.matrix((nvmx, nparams, neles), tags={'align'})}

                eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
                {'nvmax': nvmx, 'ls': ls, 'ubar': ubar, 'srafac': srafac,
                 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
                 'sigma' : sigma, 'rootrs': rootrs, 'gc': gc, 'rot': rot, 'shift': shift
                })

                eles._set_external('acteddy',
                                   f'in broadcast-col fpdtype_t[{nvmx}][{nparams}]',
                                   value=actbuff['acteddy'])

                self.actbuffs.append(actbuff)

        if not bool(self.actbuffs):
           self.tnext = math.inf
                     
    def __call__(self, intg):
        
        tcurr = intg.tcurr
        if tcurr+self.dtol >= self.tnext:
            for abid, actbuff in enumerate(self.actbuffs):    
                if actbuff['trcl'] <= self.tnext:
                    trcl = np.inf
                    for geid, xttluts in actbuff['xttlut'].items():
                        if xttluts['vid'].any():
                            #geid = actbuff['geid'][leid]
                            shft = next((i for i,v in enumerate(actbuff['buff'][:,geid]['te']) if v > tcurr),actbuff['nvmx'])
                            if shft:
                                newb = np.zeros(shft, self.buffdtype)
                                xxxx = self.xvortbuff[xttluts['vid'][:shft]]
                                pad = shft-xxxx.shape[0]
                                newb[['loci', 'ti', 'eps']] = np.pad(xxxx, (0,pad), 'constant')
                                newb[['ts', 'te']] = np.pad(xttluts[['ts', 'te']][:shft], (0,pad), 'constant')
                                self.actbuffs[abid]['buff'][:,geid] = np.concatenate((actbuff['buff'][shft:,geid],newb))
                                self.actbuffs[abid]['xttlut'][geid] = xttluts[shft:]
                                if self.actbuffs[abid]['xttlut'][geid]['vid'].any() and (self.actbuffs[abid]['buff'][-1,geid]['ts'] < trcl):
                                    trcl = self.actbuffs[abid]['buff']['ts'][-1,geid]
                    self.actbuffs[abid]['acteddy'].set(np.moveaxis(structured_to_unstructured(actbuff['buff']), 2, 1))
            self.tnext = min(etype['trcl'] for etype in self.actbuffs)
