# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t rad = ${vortrad};
  fpdtype_t invrad2 = 1.0/(${vortrad}*${vortrad});
  fpdtype_t invsigma2 = 1.0/(0.7*0.7);
  fpdtype_t xvel = ${xvel};
  fpdtype_t r2;
  fpdtype_t pos[${ndims}];
  fpdtype_t utilde[${ndims}];
  fpdtype_t eps[${ndims}];
  fpdtype_t delta[${ndims}];
  fpdtype_t arg;
  fpdtype_t magic = 1.0;
  utilde[0] = 0.0;
  utilde[1] = 0.0;
  utilde[2] = 0.0;
  % for j in range(nvmax):
    arg = 0.0;
    if (t > acteddy[${j}][5] && t < acteddy[${j}][6])
    {
        pos[0] = acteddy[${j}][0] + (t-acteddy[${j}][3])*xvel;
        pos[1] = acteddy[${j}][1];
        pos[2] = acteddy[${j}][2];
        
        % for i in range(ndims):
            arg += 0.5*invsig2*invrad2*(pos[${i}]-ploc[${i}])*(pos[${i}]-ploc[${i}]);
        % endfor

        g = ${pyfr.polyfit(lambda x: exp(x), 0, 1, 8, 'arg')};
        
        eps[0] = acteddy[${j}][4]
        eps[1] = acteddy[${j}][7]
        eps[2] = acteddy[${j}][8]
        
        % for i in range(ndims): 
            utilde[${i}] += eps[${i}]*g
        % endfor 
         
         
        src[0] += r2 < rad*rad ? acteddy[${j}][4] : 0.0;
        
        
        
        % for i in range(ndims):
            src[${i+1}] += r2 < rad*rad ? acteddy[${j}][4]*(pos[${i}]-ploc[${i}]) : 0.0;
        % endfor
        
        
        
        
        
        src[${nvars-1}] += r2 < rad*rad ? acteddy[${j}][4] : 0.0;
    }
  % endfor
</%pyfr:macro>
