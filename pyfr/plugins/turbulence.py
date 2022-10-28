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

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend

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
        e = self.cfg.getliteral(cfgsect, 'rot-axis')

        shift = np.array(c)

        qi = e[0]*np.sin(theta/2) 
        qj = e[1]*np.sin(theta/2)
        qk = e[2]*np.sin(theta/2)
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
        
        print(a21)


        rot = np.array([[a11, a12, a13],
                        [a21, a22, a23],
                        [a31, a32, a33]])

        print(rot)
        print(shift)
  
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
                    temp.append([xinit,yinit,zinit,t,epsx,epsy,epsz])
                    if periodicdim == 'y':
                        if yinit+ls>ymax:
                            temp.append([xinit,ymin-(ymax-yinit),zinit,t,epsx,epsy,epsz])
                        if yinit-ls<ymin:
                            temp.append([xinit,ymax+(yinit-ymin),zinit,t,epsx,epsy,epsz])
                    if periodicdim == 'z':
                        if zinit+ls>zmax:
                            temp.append([xinit,yinit,zmin-(zmax-zinit),t,epsx,epsy,epsz])
                        if zinit-ls<zmin:
                            temp.append([xinit,yinit,zmax+(zinit-zmin),t,epsx,epsy,epsz])
                t += (xmax-xinit)/ubar
                initial = False
            vid += 1

        self.vortbuff = np.asarray(temp)

        #####################
        # Make action buffer#
        #####################

        self.actbuffs = []

        for etype, eles in intg.system.ele_map.items():

            neles = eles.neles
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, -1)
            #print(pts.reshape(3, -1))
            #print((pts.reshape(3, -1) - shift[:,None]))
            #print(rot @ (pts.reshape(3, -1) - shift[:,None]))
            ptsr = (rot @ (pts.reshape(3, -1) - shift[:,None])).reshape(pts.shape)
            inside = bbox.pts_in_region(ptsr)

            ttlut = defaultdict(list)
            vidx = defaultdict()
            vtim = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0] # eles in injection box
                ptsri = ptsr[:,eids,:] # points in injection box
                for vid, vort in enumerate(self.vortbuff):
                    vbox = BoxRegion([vort[0]-ls, vort[1]-ls, vort[2]-ls],
                                     [xmax+ls, vort[1]+ls, vort[2]+ls])
                    elestemp = []               
                    vinside = vbox.pts_in_region(ptsri)

                    if np.any(vinside):
                        elestemp = np.any(vinside, axis=0).nonzero()[0].tolist() # injection box local indexing
                        
                    for eid in elestemp:
                        exmin = ptsri[:,eid,0].min()
                        exmax = ptsri[:,eid,0].max()
                        ts = max(vort[3], vort[3] + ((exmin - vort[0] - ls)/ubar))
                        te = ts + (exmax-exmin+2*ls)/ubar
                        ttlut[eid].append([vid,ts,te])

                for k, v in ttlut.items():
                    v.sort(key=lambda x: x[1])
                    nv = np.array(v)
                    vidx[k] = nv[:,0].astype(int)
                    vtim[k] = nv[:,-2:]

                nvmx = 0
                for leid, actl in vtim.items():
                    for i, te in enumerate(actl[:,1]):
                        shft = next((j for j,v in enumerate(actl[:,0]) if v > te+btol),len(actl)-1) - i + 1
                        if shft > nvmx:
                            nvmx = shft

                buff = np.full((nvmx, nparams, neles), 0.0)
                actbuff = {'geid': eids, 'pts': ptsr, 'trcl': 0.0, 'neles': neles, 'etype': etype, 'vidx': vidx, 'vtim': vtim, 'nvmx': nvmx, 'buff': buff, 'acteddy': eles._be.matrix((nvmx, nparams, neles), tags={'align'})}

                eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence',
                {'nvmax': nvmx, 'ls': ls, 'ubar': ubar, 'srafac': srafac,
                 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
                 'sigma' : sigma, 'rootrs': rootrs, 'gc': gc,
                 'a11': a11, 'a12': a12, 'a13': a13, 'a21': a21, 'a22': a22, 'a23': a23, 'a31': a31, 'a32': a32, 'a33': a33,
                 'cx': c[0], 'cy': c[1], 'cz': c[2]
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
                    for leid, vidxs in actbuff['vidx'].items():
                        if vidxs.any():
                            geid = actbuff['geid'][leid]
                            shft = next((i for i,v in enumerate(actbuff['buff'][:,8,geid]) if v > tcurr),actbuff['nvmx'])
                            if shft:
                                newb = np.concatenate((self.vortbuff[vidxs[:shft],:], actbuff['vtim'][leid][:shft,:]), axis=1)
                                newb = np.pad(newb, [(0,shft-newb.shape[0]),(0,0)], 'constant')
                                self.actbuffs[abid]['buff'][:,:,geid] = np.concatenate((actbuff['buff'][shft:,:,geid],newb))
                                self.actbuffs[abid]['vidx'][leid] = vidxs[shft:]
                                self.actbuffs[abid]['vtim'][leid] = actbuff['vtim'][leid][shft:,:]
                                if self.actbuffs[abid]['vidx'][leid].any() and (self.actbuffs[abid]['buff'][-1,7,geid] < trcl):
                                    trcl = self.actbuffs[abid]['buff'][-1,7,geid]
                    self.actbuffs[abid]['trcl'] = trcl
                    self.actbuffs[abid]['acteddy'].set(actbuff['buff'])
            self.tnext = min(etype['trcl'] for etype in self.actbuffs)
