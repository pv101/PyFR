# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t rad = ${vortrad};
  fpdtype_t invrad2 = 1.0/(${vortrad}*${vortrad});
  fpdtype_t invsigma2 = 1.0/(0.7*0.7);
  fpdtype_t invsigma = 1.0/0.7;
  fpdtype_t xvel = ${xvel};
  fpdtype_t pos[${ndims}];
  fpdtype_t utilde[${ndims}];
  fpdtype_t eps[${ndims}];
  fpdtype_t delta2[${ndims}];
  fpdtype_t arg;
  fpdtype_t magic = 1.0;
  fpdtype_t rs = 0.001;
  utilde[0] = 0.0;
  utilde[1] = 0.0;
  utilde[2] = 0.0;
  fpdtype_t xloc2;
  fpdtype_t clip;
  fpdtype_t g;
  fpdtype_t xin = 0.5;
  fpdtype_t srafac = 0.007075599999999999;
  
  % for j in range(nvmax):
    arg = 0.0;
    if (t > acteddy[${j}][5] && t < acteddy[${j}][6])
    {
        pos[0] = acteddy[${j}][0] + (t-acteddy[${j}][3])*xvel;
        pos[1] = acteddy[${j}][1];
        pos[2] = acteddy[${j}][2];
        
        % for i in range(ndims):
            delta2[${i}] = (pos[${i}]-ploc[${i}])*(pos[${i}]-ploc[${i}]);
            arg += 0.5*invsigma2*invrad2*delta2[${i}];
        % endfor
        g = delta2[0] < rad*rad ? delta2[1] < rad*rad ? delta2[2] < rad*rad ? invsigma*invsigma*invsigma*magic*magic*magic*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0 : 0.0 : 0.0;
        
        eps[0] = acteddy[${j}][4];
        eps[1] = acteddy[${j}][7];
        eps[2] = acteddy[${j}][8];
        
        % for i in range(ndims): 
            utilde[${i}] += eps[${i}]*g;
        % endfor     
    }
  % endfor
  
  % for i in range(ndims): 
    utilde[${i}] *= rs;
  % endfor
  
  xloc2 = -0.5*3.141592654*(xin-ploc[0])*(xin-ploc[0])*invrad2;
  clip = ${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')};
  
  src[0] += srafac*utilde[0]*(xvel/rad)*clip;
  
  % for i in range(ndims):
    src[${i+1}] += u[0]*utilde[${i}]*(xvel/rad)*clip;
  % endfor
  
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};
        
  src[${nvars-1}] += 0.5*u[0]*udotu_fluct*(xvel/rad)*clip;
  
</%pyfr:macro>
