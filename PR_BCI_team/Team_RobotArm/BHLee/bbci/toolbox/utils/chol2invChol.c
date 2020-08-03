#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  double *invL , *L;
  int n, i, j, q;
  char *Lower = "L"; /* Lower triagonal matrix */
  char *N = "N";
  
  L = mxGetPr(prhs[0]);
  n = mxGetN(prhs[0]);
  
  if (nrhs != 1 || nlhs > 1){
    mexErrMsgTxt("Usage: invL = chol2invChol(L)");
  }  
  
  /* Compute the inverse of the lower triagonal matrix L */
  plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
  invL = mxGetPr(plhs[0]);
  
  memcpy(invL,L,n*n*sizeof(double));
  dtrtri_(Lower, N, &n, invL, &n, &q);
  
  for (i = 0; i<n; i++) {
    for (j = i+1; j<n; j++){
        invL[i+j*n] = 0;
    }
  }
  
}
