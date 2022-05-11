# -*- coding: utf-8 -*-

import numpy as np

from pyfr.plugins.base import BasePlugin

class Source(BasePlugin):
    name = 'source'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        for etype, eles in intg.system.ele_map.items():
            eles.add_source_macro('pyfr.plugins.kernels.source','source',{})

    def __call__(self, intg):
        pass
