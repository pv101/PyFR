# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t xin = ${xin};
  fpdtype_t ls = ${ls};
  fpdtype_t ls2 = ${ls}*${ls};
  fpdtype_t invls2 = 1.0/(${ls}*${ls});
  fpdtype_t gc3 = ${gc}*${gc}*${gc};
  fpdtype_t rootrs = ${rootrs};
  fpdtype_t srafac = ${srafac};
  fpdtype_t invsigma = 1.0/${sigma};
  fpdtype_t invsigma2 = 1.0/(${sigma}*${sigma});
  fpdtype_t invsigma3 = 1.0/(${sigma}*${sigma}*${sigma});
  fpdtype_t ubar = ${ubar};
  fpdtype_t pos[${ndims}];
  fpdtype_t tploc[${ndims}];
  fpdtype_t ttploc[${ndims}];
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
  fpdtype_t a11 = ${a11};
  fpdtype_t a12 = ${a12};
  fpdtype_t a13 = ${a13};
  fpdtype_t a21 = ${a21};
  fpdtype_t a22 = ${a22};
  fpdtype_t a23 = ${a23};
  fpdtype_t a31 = ${a31};
  fpdtype_t a32 = ${a32};
  fpdtype_t a33 = ${a33};
  fpdtype_t cx = ${cx};
  fpdtype_t cy = ${cy};
  fpdtype_t cz = ${cz};
  
  % for i in range(nvmax):
    arg = 0.0;
    if (t > acteddy[${i}][7] && t < acteddy[${i}][8])
    {
        pos[0] = acteddy[${i}][0] + (t-acteddy[${i}][3])*ubar;
        pos[1] = acteddy[${i}][1];
        pos[2] = acteddy[${i}][2];

        tploc[0]=ploc[0]+cx;
        tploc[1]=ploc[1]+cy;
        tploc[2]=ploc[2]+cz;

        ttploc[0] = a11*tploc[0] + a12*tploc[1] + a13*tploc[2];
        ttploc[1] = a21*tploc[0] + a22*tploc[1] + a23*tploc[2];
        ttploc[2] = a31*tploc[0] + a32*tploc[1] + a33*tploc[2];
        
        % for j in range(ndims):
            delta2[${j}] = (pos[${j}]-ttploc[${j}])*(pos[${j}]-ttploc[${j}]);
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
    utilde[${i}] *= rootrs;
  % endfor
  
  xloc2 = -0.5*3.141592654*(xin-ttploc[0])*(xin-ttploc[0])*invls2;
  
  clipx = ttploc[0] < xmax ? ttploc[0] > xmin ? ${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0: 0.0;
  clipy = ttploc[1] < ymax ? ttploc[1] > ymin ? 1.0 : 0.0: 0.0;
  clipz = ttploc[2] < zmax ? ttploc[2] > zmin ? 1.0 : 0.0: 0.0;
  
  clip = clipx;
  
  src[0] += srafac*utilde[0]*(ubar/ls)*clip;
  
  % for i in range(ndims):
    src[${i+1}] += u[0]*utilde[${i}]*(ubar/ls)*clip;
  % endfor
  
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};
        
  src[${nvars-1}] += 0.5*u[0]*udotu_fluct*(ubar/ls)*clip;
  
</%pyfr:macro>
