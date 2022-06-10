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
    if (t > acteddy[${j}][4] && t < acteddy[${j}][5])
    {
        % for i in range(ndims):
            r2 += (acteddy[${j}][${i}]-ploc[${i}])*(acteddy[${j}][${i}]-ploc[${i}]);
        % endfor
        
        src[0] += r2 < rad*rad ? acteddy[${j}][3] : 0.0;
        % for i in range(ndims):
            src[${i+1}] += r2 < rad*rad ? acteddy[${j}][3]*(acteddy[${j}][${i}]-ploc[${i}]) : 0.0;
        % endfor
        src[${nvars-1}] += r2 < rad*rad ? acteddy[${j}][3] : 0.0;
    }
  % endfor
</%pyfr:macro>
