# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  % for i in range(nvars):
    src[${i}] += ${nvmax};
  % endfor
</%pyfr:macro>
