# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.baseadvec.kernels.vort'/>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              acteddy='in broadcast-col fpdtype_t[${str(nvmax)}][${str(nvpar)}]'
              rcpdjac='in fpdtype_t'>
              
fpdtype_t stemp[${nvars}] = {};

for(int i =0; i<${nvmax}; i++)
{
    fpdtype_t eloc[] = ${pyfr.array(acteddy[i][{j}], j=(0, ndims))};
    fpdtype_t rad = acteddy[i][${ndims}];
    fpdtype_t eps = acteddy[i][${ndims+1}];
    ${pyfr.expand('vort', 'rad', 'eps', 'eloc', 'ploc', 'stemp')};
}              
                         
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex} + stemp[${i}];
% endfor

</%pyfr:kernel>
