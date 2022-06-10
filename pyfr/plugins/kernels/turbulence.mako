# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t rad = ${vortrad};
  fpdtype_t xvel = ${xvel};
  fpdtype_t r2;
  fpdtype_t pos[${ndims}];
  % for j in range(nvmax):
    r2 = 0.0;
    if (t > acteddy[${j}][5] && t < acteddy[${j}][6])
    {
        pos[0] = acteddy[${j}][0] + (t-acteddy[${j}][3])*xvel;
        pos[1] = acteddy[${j}][1];
        
        % if ndims == 3:
            pos[2] = acteddy[${j}][2];
        % endif
        
        % for i in range(ndims):
            r2 += (pos[${i}]-ploc[${i}])*(pos[${i}]-ploc[${i}]);
        % endfor
        
        src[0] += r2 < rad*rad ? acteddy[${j}][4] : 0.0;
        % for i in range(ndims):
            src[${i+1}] += r2 < rad*rad ? acteddy[${j}][4]*(pos[${i}]-ploc[${i}]) : 0.0;
        % endfor
        src[${nvars-1}] += r2 < rad*rad ? acteddy[${j}][4] : 0.0;
    }
  % endfor
</%pyfr:macro>
