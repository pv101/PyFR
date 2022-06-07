# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  % for i in range(nvars):
    src[${i}] += 0.0;
  % endfor
  fpdtype_t rad = ${vortrad};
  fpdtype_t r2;
  % for j in range(nvmax):
    r2 = 0.0;
    % for i in range(ndims):
        r2 += (acteddy[${j}][${i}]-ploc[${i}])*(acteddy[${j}][${i}]-ploc[${i}]);
    % endfor
    src[0] += r2 < rad*rad ? acteddy[${j}][2] : 0.0;
    src[1] += r2 < rad*rad ? acteddy[${j}][2]*(acteddy[${j}][1]-ploc[1]) : 0.0;
    src[2] += r2 < rad*rad ? -acteddy[${j}][2]*(acteddy[${j}][0]-ploc[0]) : 0.0;
    src[3] += r2 < rad*rad ? acteddy[${j}][2] : 0.0;
  % endfor
</%pyfr:macro>
