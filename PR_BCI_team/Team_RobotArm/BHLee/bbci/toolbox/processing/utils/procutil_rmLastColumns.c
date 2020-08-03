/*
 * function procutil_rmLastColumns
 *
 * Usage:
 *          procutil_rmLastColumns(A, colToRemove)
 *
 * Parameters:
 *                A - a n x m matrix with n rows and m columns
 *      colToRemove - The number of col to remove from the matrix
 *
 * Description:
 *          The function removes the last colToRemove columns from the Matrix 
 *          A. If A is a n x m matrix with n rows and m columns. Then A will
 *          be a n x (m - colToRemove) matrix after the function call.
 *
 * 2009/10/27 - Max Sagebaum
 *          - file created
 * 2009/10/30 - Max Sagebaum
 *          - added call to unshare the matrix array, there were some problems
 *            with: A = rand(1000,1000)
 *                  B = A;
 *                  procutil_rmLastColumns(A,100)
 *            The problem was a memory access violation.
 *
 * (c) 2009 Fraunhofer FIRST 
 */

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int rmLinesCount;
    double* dataPointer;
    int m,n;
    
    /* check the conditions for the function */
    /* only 2 arguments */
    if(2 != nrhs) {
        mexErrMsgTxt("proc_rmLastChannel: two arguments required!");
        return;
    }
    /* no sparse matrix */
    if(mxIsSparse(prhs[0])) {
        mexErrMsgTxt("proc_rmLastChannel: only non sparse matrix supported!");
        return;
    }
    
    /* no complex matrix */
    if(mxIsComplex(prhs[0])) {
        mexErrMsgTxt("proc_rmLastChannel: only real valued matrices are allowed!");
        return;
    }
    
    /* only double matrix */
    if(!mxIsDouble(prhs[0])) {
        mexErrMsgTxt("proc_rmLastChannel: only double valued matrices are allowed!");
        return;
    }
    
    /* only double valued non complex remove lines */ 
    if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("proc_rmLastChannel: second argument is not a double number!");
        return;
    }
    
    rmLinesCount = (int)mxGetScalar(prhs[1]);
    
    /* columns of the matrix must be greater the columns to remove */
    if(mxGetN(prhs[0]) < rmLinesCount) {
        mexErrMsgTxt("proc_rmLastChannel: number of columns to remove is larger than the number of columns in the matrix!");
        return;
    }
    
    /* first unshare the matrix, ohterwise there will be invalid acces exceptions */
    /* mxUnshareArray is an undocumented funciton */
    mxUnshareArray(prhs[0],true);
    
    /* get the values from the matrix */
    n = mxGetN(prhs[0]); /* columns */
    m = mxGetM(prhs[0]);
    dataPointer = mxGetPr(prhs[0]);
    
    /* free the memory */
    dataPointer = mxRealloc(dataPointer, m * (n - rmLinesCount) * sizeof(*dataPointer));
    
    /* set the new values in the matrix */
    mxSetN((mxArray*)prhs[0], n - rmLinesCount);
    mxSetPr((mxArray*)prhs[0], dataPointer);
}
