# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t xin = 0.0;
  fpdtype_t ls = ${ls};
  fpdtype_t ls2 = ${ls*ls};
  fpdtype_t invls2 = ${1.0/(ls*ls)};
  fpdtype_t gc3 = ${gc*gc*gc};
  fpdtype_t rootrs = ${rootrs};
  fpdtype_t srafac = ${srafac};
  fpdtype_t invsigma2 = ${1.0/(sigma*sigma)};
  fpdtype_t invsigma3 = ${1.0/(sigma*sigma*sigma)};
  fpdtype_t ubar = ${ubar};
  fpdtype_t pos[${ndims}];
  fpdtype_t posper[2][${ndims}];
  fpdtype_t ttploc[${ndims}];
  fpdtype_t eps[${ndims}];
  fpdtype_t delta2[${ndims}];
  fpdtype_t arg;
  fpdtype_t utilde[${ndims}];
  utilde[0] = 0.0;
  utilde[1] = 0.0;
  utilde[2] = 0.0;
  fpdtype_t xloc2;
  fpdtype_t sclip;
  fpdtype_t g;
  fpdtype_t xmin = -${ls};
  fpdtype_t xmax = ${ls};
  fpdtype_t ymin = ${ymin};
  fpdtype_t ymax = ${ymax};
  fpdtype_t zmin = ${zmin};
  fpdtype_t zmax = ${zmax};
  fpdtype_t fac = ${-0.5/(sigma*sigma*ls*ls)};
  fpdtype_t fac2 = ${(gc*gc*gc)/(sigma*sigma*sigma)};
  
  fpdtype_t tbc = 2.3283064365386962890625e-10;
  
  uint32_t oldstate;
  uint32_t newstate;
  uint8_t rshift;
  uint8_t b22 = 22;
  uint8_t b32 = 32;
  uint8_t opbits = 4;

  //fpdtype_t delsigny;
  fpdtype_t delsignz;

  int epscomp;
  
  % for i, r in enumerate(rot):
    ttploc[${i}] = ${' + '.join(f'{r[j]}*(ploc[{j}] - {shift[j]})' for j in range(3))};
  % endfor

  % for i in range(nvmax):
    posper[0][0] = xmin + (t-tinit[${i}][0])*ubar;
    
    oldstate = state[${i}][0];
    newstate = (oldstate * 747796405UL) + 2891336453UL;
    rshift = oldstate >> (b32 - opbits);
    oldstate ^= oldstate >> (opbits + rshift);
    oldstate *= 277803737UL;
    oldstate ^= oldstate >> b22;
    posper[0][1] = ymin + (ymax-ymin)*((fpdtype_t)oldstate * tbc);
    
    oldstate = newstate;
    newstate = (oldstate * 747796405UL) + 2891336453UL;
    rshift = oldstate >> (b32 - opbits);
    oldstate ^= oldstate >> (opbits + rshift);
    oldstate *= 277803737UL;
    oldstate ^= oldstate >> b22;
    posper[0][2] = zmin + (zmax-zmin)*((fpdtype_t)oldstate * tbc);
    
    oldstate = newstate;
    newstate = (oldstate * 747796405UL) + 2891336453UL;
    rshift = oldstate >> (b32 - opbits);
    oldstate ^= oldstate >> (opbits + rshift);
    oldstate *= 277803737UL;
    oldstate ^= oldstate >> b22;
    epscomp = oldstate % 8;
    
    //delsigny = posper[1][0] < 0.5*(ymax-ymin) ? (ymax-ymin) : -(ymax-ymin);
    
    delsignz = posper[2][0] < 0.5*(zmax-zmin) ? (zmax-zmin) : -(zmax-zmin);
    
    posper[1][0] = posper[0][0];
    posper[1][1] = posper[0][1];
    posper[1][2] = posper[0][2] + delsignz;
    
    //posper[0][1] = posper[0][0];
    //posper[0][2] = posper[0][0];
    //posper[0][3] = posper[0][0];
    
    //posper[1][1] = posper[1][0] + delsigny;
    //posper[1][2] = posper[1][0];
    //posper[1][3] = posper[1][0] + delsigny;
    
    //posper[2][1] = posper[2][0];
    //posper[2][2] = posper[2][0] + delsignz;
    //posper[2][3] = posper[2][0] + delsignz;
    
    eps[0] = (epscomp & 1) ? -1 : 1;
    eps[1] = (epscomp & 2) ? -1 : 1;
    eps[2] = (epscomp & 4) ? -1 : 1;
    
    % for k in range(2): 
      arg = 0.0;
      % for j in range(ndims):
        delta2[${j}] = (posper[${k}][${j}]-ttploc[${j}])*(posper[${k}][${j}]-ttploc[${j}]);
        arg += fac*delta2[${j}];
      % endfor

      g = delta2[0] < ls2 ? delta2[1] < ls2 ? delta2[2] < ls2 ? posper[0][0] <= xmax ? fac2*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0 : 0.0 : 0.0 : 0.0;

      % for j in range(ndims): 
        utilde[${j}] += eps[${j}]*g;
      % endfor
    % endfor
    
  % endfor
  
  % for i in range(ndims): 
    utilde[${i}] *= rootrs;
  % endfor
  
  xloc2 = -0.5*3.141592654*ttploc[0]*ttploc[0]*invls2;
  
  sclip = ttploc[0] < xmax ? ttploc[0] > xmin ? (ubar/ls)*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0: 0.0;
  
  src[0] += srafac*utilde[0]*sclip;
  % for i in range(ndims):
    src[${i+1}] += u[0]*utilde[${i}]*sclip;
  % endfor
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};     
  src[${nvars-1}] += 0.5*u[0]*udotu_fluct*sclip;
  
</%pyfr:macro>
