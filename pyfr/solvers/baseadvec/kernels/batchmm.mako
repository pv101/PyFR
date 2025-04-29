<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

## v = A*u
<%pyfr:kernel name='batchmm' ndim='1'
              A='in fpdtype_t[${str(na)}][${str(nb)}]'
              u='in fpdtype_t[${str(nb)}][${str(nvars)}]'
              v='out fpdtype_t[${str(na)}][${str(nvars)}]'>
    for (int i = 0; i < ${nvars}; i++)
        for (int j = 0; j < ${na}; j++)
        {
            fpdtype_t tmp = 0;

            for (int k = 0; k < ${nb}; k++)
                tmp += A[j][k]*u[k][i];

            v[j][i] = tmp;
        }
</%pyfr:kernel>
