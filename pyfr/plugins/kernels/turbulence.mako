# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t xin = ${xin};
  fpdtype_t ls = ${ls};
  fpdtype_t ls2 = ${ls}*${ls};
  fpdtype_t invls2 = 1.0/(${ls}*${ls});
  fpdtype_t gc3 = ${gc}*${gc}*${gc};
  fpdtype_t rs = ${rs};
  fpdtype_t srafac = ${srafac};
  fpdtype_t invsigma = 1.0/${sigma};
  fpdtype_t invsigma2 = 1.0/(${sigma}*${sigma});
  fpdtype_t invsigma3 = 1.0/(${sigma}*${sigma}*${sigma});
  fpdtype_t ubar = ${ubar};
  fpdtype_t pos[${ndims}];
  fpdtype_t eps[${ndims}];
  fpdtype_t delta2[${ndims}];
  fpdtype_t arg;
  fpdtype_t utilde[${ndims}];
  utilde[0] = 0.0;
  utilde[1] = 0.0;
  utilde[2] = 0.0;
  fpdtype_t xloc2;
  fpdtype_t clip;
  fpdtype_t clipx;
  fpdtype_t clipy;
  fpdtype_t clipz;
  fpdtype_t g;
  fpdtype_t xmin = ${xin} - ${ls};
  fpdtype_t xmax = ${xin} + ${ls};
  fpdtype_t ymin = ${ymin};
  fpdtype_t ymax = ${ymax};
  fpdtype_t zmin = ${zmin};
  fpdtype_t zmax = ${zmax};
  
  % for i in range(nvmax):
    arg = 0.0;
    if (t > acteddy[${i}][7] && t < acteddy[${i}][8])
    {
        pos[0] = acteddy[${i}][0] + (t-acteddy[${i}][3])*ubar;
        pos[1] = acteddy[${i}][1];
        pos[2] = acteddy[${i}][2];
        
        % for j in range(ndims):
            delta2[${j}] = (pos[${j}]-ploc[${j}])*(pos[${j}]-ploc[${j}]);
            arg += -0.5*invsigma2*invls2*delta2[${j}];
        % endfor
        g = delta2[0] < ls2 ? delta2[1] < ls2 ? delta2[2] < ls2 ? invsigma3*gc3*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0 : 0.0 : 0.0;
        
        eps[0] = acteddy[${i}][4];
        eps[1] = acteddy[${i}][5];
        eps[2] = acteddy[${i}][6];
        
        % for j in range(ndims): 
            utilde[${j}] += eps[${j}]*g;
        % endfor     
    }
  % endfor
  
  % for i in range(ndims): 
    utilde[${i}] *= rs;
  % endfor
  
  xloc2 = -0.5*3.141592654*(xin-ploc[0])*(xin-ploc[0])*invls2;
  
  clipx = ploc[0] < xmax ? ploc[0] > xmin ? ${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0: 0.0;
  clipy = ploc[1] < ymax ? ploc[1] > ymin ? 1.0 : 0.0: 0.0;
  clipz = ploc[2] < zmax ? ploc[2] > zmin ? 1.0 : 0.0: 0.0;
  
  clip = clipx*clipy*clipz;
  
  src[0] += srafac*utilde[0]*(ubar/ls)*clip;
  
  % for i in range(ndims):
    src[${i+1}] += u[0]*utilde[${i}]*(ubar/ls)*clip;
  % endfor
  
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};
        
  src[${nvars-1}] += 0.5*u[0]*udotu_fluct*(ubar/ls)*clip;
  
</%pyfr:macro>
