# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin

class Source(BasePlugin):
    name = 'source'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        nvmax, q = 10, 10
        
        self.acteddy = acteddy = {}

        for etype, eles in intg.system.ele_map.items():
            eles.add_source_macro('pyfr.plugins.kernels.source','source', {'nvmax': nvmax})
            acteddy[etype] = self._be.matrix((nvmax, q, eles.neles), tags={'align'})
        	self._set_external('acteddy',
                               f'in broadcast-col fpdtype_t[{nvmax}][{q}]',
                               value=acteddy[etype])

    def __call__(self, intg):
        pass
