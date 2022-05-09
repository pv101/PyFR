# -*- coding: utf-8 -*-

import numpy as np
import os
import quads as qt
import random

class SynEddy(BasePlugin):
    name = 'syneddy'
    systems = ['navier-stokes']
    formulations = ['std']
    # can we restrict to 3D?

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # MPI info
        _, self.rank, self.root = get_comm_rank_root()

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Update frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 1)

        self.mesh = intg.system.mesh

        self.acteddy = self._be.matrix((nvmax, nvpar, neles), tags={'align'})

        self._set_external('acteddy',
                           'in broadcast-col fpdtype_t[1][{}]'.format(),
                           value=self.acteddy)

        self.vorts = Vorts(0.0,20,1.0,[-1,-1,-1],[1,1,1],10,10,self.mesh,self.acteddy)

        # Update the backend
        self._update_backend()
        
    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            # Update the eddies
            self.vorts.update(intg.tcurr)

            # Update the backend
            self._update_backend()

class Vorts:
	def __init__(self, tmint, nvorts, xvel, boxbl, boxtr, ny, nz, mesh, acteddy):
		
		self.nvorts = nvorts
		self.xvel = xvel

		self.acteddy = acteddy

		# vortex box

		self.boxbl = boxbl
		self.boxtr = boxtr

		# number of y and z divisions

		self.ny = ny
		self.nz = nz

		# dimensions in y and z

		self.ymin = boxbl[1]
		self.ymax = boxtr[1]

		self.zmin = boxbl[2]
		self.zmax = boxtr[2]

		self.ydim = boxtr[1]-boxbl[1]
		self.zdim = boxtr[2]-boxbl[2]

		self.dy = (boxtr[1]-boxbl[1])/ny
		self.dz = (boxtr[2]-boxbl[2])/nz

		self.yzcentre = ((boxtr[1]+boxbl[1])/2,(boxtr[2]+boxbl[2])/2)

		# quad tree in y and z

		self.tree = qt.QuadTree(self.yzcentre, self.ydim, self.zdim)

		# division centres in y and z

		self.ycents = np.linspace(self.ymin+self.dy/2,self.ymax-self.dy/2,self.ny)
		self.zcents = np.linspace(self.zmin+self.dz/2,self.zmax-self.dz/2,self.nz)

		# fill quad tree

		for i in range(self.ny):
			for j in range(self.nz):
				qboxbl = [boxbl[0],self.ycents[i]-self.dy/2,self.zcents[j]-self.dz/2]
				qboxbl = [boxtr[0],self.ycents[i]+self.dy/2,self.zcents[j]+self.dz/2]

				cells = []

				for:
					cells.append((eidx, etype, xmin, xmax))
				
				self.tree.insert((self.ycents[i], self.zcents[j]), data=cells)

		# setup vortices

		self.vorts = []

		for i in range(self.nvorts):
			self.vorts.append(Vort(tmint, i, self.xvel, self.boxbl, self.boxtr, self.ny, self.nz, self.tree, self.acteddy))

	def update(self, tcurr):
		for vort in self.vorts:
			vort.update(tcurr)

class Vort:
	def __init__(self, tmint, vortid, xvel, boxbl, boxtr, ny, nz, tree, acteddy):
		self.vortid = vortid
		self.xvel = xvel
		self.tmint = tmint

		self.acteddy = acteddy

		# vortex box

		self.boxbl = boxbl
		self.boxtr = boxtr

		# number of y and z divisions

		self.ny = ny
		self.nz = nz

		# dimensions in y and z

		self.ymin = boxbl[1]
		self.ymax = boxtr[1]

		self.zmin = boxbl[2]
		self.zmax = boxtr[2]

		self.ydim = boxtr[1]-boxbl[1]
		self.zdim = boxtr[2]-boxbl[2]

		self.dy = (boxtr[1]-boxbl[1])/ny
		self.dz = (boxtr[2]-boxbl[2])/nz

		self.yzcentre = ((boxtr[1]+boxbl[1])/2,(boxtr[2]+boxbl[2])/2)
		
		self.radius = 0
		self.eps = 0
		self.x = 0
		self.y = 0
		self.z = 0

		self.tree = tree

		self.eles = []
		self.geteles()

	def geteles(self):
		self.radius = random.uniform(0.02, 0.07)
		self.eps = random.uniform(0.8, 1.0)
		self.x = self.xmin + self.radius
		self.y = random.uniform(self.ymin+self.radius+0.1, self.ymax-self.radius-0.1)
		self.z = random.uniform(self.zmin+self.radius+0.1, self.zmax-self.radius-0.1)
		
		self.eles.clear()

		yminv = self.y-self.radius-self.dy/2
		ymaxv = self.y+self.radius+self.dy/2
		zminv = self.z-self.radius-self.dz/2
		zmaxv = self.z+self.radius+self.dz/2

		# define bb for vortex
		bb = qt.BoundingBox(min_x=yminv, min_y=zminv, max_x=ymaxv, max_y=zmaxv)

		# get quads in bb
		tiles = self.tree.within_bb(bb)

		eles = set() # a set

		# get eles from tiles and add to set of eles
		for tile in tiles:
			for ele in tile.data:
				eles.add(ele)

		for ele in eles:
			eidx = ele[0]
			etype = ele[1]
			exmin = ele[2]
			exmax = ele[3]
			tpush = max(self.tstart, self.tstart + (exmin - (self.xmin + 2*self.radius))/self.xvel)
			tpop = tpush + (exmax-exmin+2*self.radius)/self.xvel
			self.eles.append({"eidx": eidx, "etype": etype, "type": "add", "t": tpush})
			self.eles.append({"eidx": eidx, "etype": etype, "type": "remove", "t": tpop})
		
		self.eles.sort(key=lambda x: x["t"])
	
	def update(self, tcurr):
		while self.eles:
			if tcurr >= self.eles[0]["t"]:
				act = self.eles.pop(0)
				if act["type"] == "add":

					acteddy[][0][act["eidx"]] = self.x
					acteddy[][1][act["eidx"]] = self.y
					acteddy[][2][act["eidx"]] = self.z
					acteddy[][3][act["eidx"]] = self.radius
					acteddy[][4][act["eidx"]] = self.eps


				elif act["type"] == "remove":


			else:
				break

			if not self.eles:
				self.tmint = tcurr
				self.geteles()

			self.x = self.xmin + self.radius + (tcurr-self.tmint)*self.xvel