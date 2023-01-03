# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t xin = 0.0;
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
  fpdtype_t sclip;
  fpdtype_t g;
  fpdtype_t xmin = -${ls};
  fpdtype_t xmax = ${ls};
  fpdtype_t ymin = ${ymin};
  fpdtype_t ymax = ${ymax};
  fpdtype_t zmin = ${zmin};
  fpdtype_t zmax = ${zmax};
  fpdtype_t fac = -0.5*invsigma2*invls2;
  fpdtype_t fac2 = invsigma3*gc3;

  % for i, r in enumerate(rot):
    ttploc[${i}] = ${' + '.join(f'{r[j]}*(ploc[{j}] - {shift[j]})' for j in range(3))};
  % endfor

  int i;
  for (int i = 0; i < ${nvmax}; i++)
  {
    //if (acteddy[i][7] > t)
    //{
    //  break;
    //}
    //else if (acteddy[i][8] > t)
    //{
      pos[0] = acteddy[i][0] + (t-acteddy[i][3])*ubar;
      pos[1] = acteddy[i][1];
      pos[2] = acteddy[i][2];
      
      //pos[0] = 0.5 + (t-0.0)*ubar;
      //pos[1] = 0.5;
      //pos[2] = 0.5;

      arg = 0.0;
      % for j in range(ndims):
        delta2[${j}] = (pos[${j}]-ttploc[${j}])*(pos[${j}]-ttploc[${j}]);
        arg += fac*delta2[${j}];
      % endfor

      g = delta2[0] < ls2 ? delta2[1] < ls2 ? delta2[2] < ls2 ? fac2*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0 : 0.0 : 0.0;
      
      //g = 1.0;
      
      eps[0] = acteddy[i][4];
      eps[1] = acteddy[i][5];
      eps[2] = acteddy[i][6];
      
      % for j in range(ndims): 
        utilde[${j}] += eps[${j}]*g;
      % endfor
    //}
  }
  
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
