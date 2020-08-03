/* inv_logdet_pd - a function to compute the inverse of a square, positive
   definite symmetric matrix, and (optionally) the log of the determinant. This
   functin is twice as fast as matlabs builtin "inv" function. 

   Additions by Anton Schwaighofer: Now allowing third return argument to 
   check whether the matrix is positive definite

   Copyright (c) 2003, Carl Edward Rasmussen. 30-12-2003. */

#include "mex.h"
#include <math.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *A, *C, z = 0.0, *Z;
  int n, i, j, q;
  char *U = "U";
  A = mxGetPr(prhs[0]);
  n = mxGetN(prhs[0]);
  if (nrhs != 1 || nlhs > 3)
    mexErrMsgTxt("Usage: [invA, logdetA, q] = inv_logdet_pd(A)");
  if (n != mxGetM(prhs[0]))
    mexErrMsgTxt("Error: Argument matrix must be square");
  n = mxGetN(prhs[0]);
  plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
  if (nlhs > 1)
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  C = mxGetPr(plhs[0]);
  memcpy(C,A,n*n*sizeof(double));
  dpotrf_(U, &n, C, &n, &q);                /* cholesky */
  if (nlhs > 2) {
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    Z = mxGetPr(plhs[2]);
    Z[0] = (double) q;
  }
  if ((q > 0) && (nlhs < 3))
    mexErrMsgTxt("Error: Argument matrix must be positive definite");
  if (!(q > 0)) {
    if (nlhs > 1) {                       /* compute log det */
      Z = mxGetPr(plhs[1]);
      for (i=0; i<n; i++) z += log(C[i*(n+1)]);
      Z[0] = 2.0*z;
    }
    dpotri_(U, &n, C, &n, &q);            /* cholesky to inverse */
    for (i=0; i<n; i++) for (j=i+1; j<n; j++) C[j+i*n] = C[i+j*n];
  }
}
