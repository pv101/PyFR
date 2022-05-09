# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='vort' params='rad, eps, eloc, ploc, s'>
    fpdtype_t stemp = eps;
    fpdtype_t r2 = 0;
    % for i in range(ndims):
        r2 += (eloc[i]-ploc[i])*(eloc[i]-ploc[i])
    % endfor
    stemp *= r2 < rad*rad ? 1.0 : 0.0;
    % for i in range(nvars):
        s[${i}] = stemp;
    % endfor
</%pyfr:macro>
