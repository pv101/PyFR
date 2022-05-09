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
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
% endfor

fpdtype_t stemp[${nvars}];
% for i in range(nvmax):
    fpdtype_t eloc[] = ${pyfr.array('acteddy[${i}][${j}]', j=(0, ndims-1))};
    ${pyfr.expand('vort', acteddy[${i}][${ndims}], acteddy[${i}][${ndims+1}], eloc, ploc, 'stemp')};
    % for j in range(nvars):
        tdivtconf[${j}] += stemp[${j}];
    % endfor
% endfor
</%pyfr:kernel>
