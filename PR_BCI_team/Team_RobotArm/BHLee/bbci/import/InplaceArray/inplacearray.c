/*************************************************************************
 * function B = inplacearray(A, offset, sz)
 *
 * B = inplacearray(A)
 *  Return the inplace-array pointed to same place A
 *
 * B = inplacearray(A, OFFSET)
 *  B(1) is pointed at A(1+offset) (linear indexing)
 *   
 * B = inplacearray(A, OFFSET, SZ)
 *  Specify the dimesnion of B
 *  Alternate calling syntax B = inplacearray(A, OFFSET, N1, N2, ... Nn)
 *
 * INPUTS
 *  A; is a (full) array
 *  OFFSET: scalar, offset from the first element A(1). Note that
 *          overflow/negative value is allowed. OFFSET is 0 by default
 *  SZ: the dimension of the inplace output, overflow allowed (!).
 * OUTPUT
 *  B: nd-array of the size SZ, shared the same data than A
 *       B(1) is started from A(1+offset).
 *
 * Class supported: all numerical, logical and char
 *
 * Important notes:
 * - For advanced users only!!!! In any case use at your own risk
 * - use MEX function releaseinplace(B) to release properly shared-data
 *   pointer before clear/reuse B.
 * - All inplace variables shared data with A must be released before
 *   the original array A is cleared/reused.
 *
 * Compilation:
 *  >> buildInternal_mxArrayDef('Internal_mxArray.h')
 *  >> mex -O -v inplacearray.c % add -largeArrayDims on 64-bit computer
 *
 * Author Bruno Luong <brunoluong@yahoo.com>
 * Last update: 27/June/2009
 * 
 ************************************************************************/

#include "mex.h"
#include "matrix.h"

/* Uncomment this on older Matlab version where size_t has not been
 * defined */
/*
 * #define mwSize int
 * #define size_t int
 */

/* The following file defines the internal representation of mxArray,
 * inspired from mxArray_tag declared in the header <matrix.h>.
 * This file is built by calling the MATLAB function
 * buildInternal_mxArrayDef.m */
#include "Internal_mxArray.h"

#define AMAT prhs[0]
#define OFFSET prhs[1]
#define SZ prhs[2]
#define BMAT plhs[0]

/* Gateway of inplacearray */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    
    mwSize M, N;
    double *dblPr;
    char *Pr, *Pi;
    mwSize nd;
    mwIndex offset;
    mwSize *dims;
    int i, freedims, flag;
    mxComplexity ComplexFlag;
    size_t szelemclass;
    mxClassID ClassID;
    
    /* Check arguments */
    if (nrhs<1)
        mexErrMsgTxt("INPLACEARRAY: At least one argument is required.");
    
    /* Size of elements in the class */
    ClassID = mxGetClassID(AMAT);
    switch (ClassID) {
        case mxDOUBLE_CLASS:
        case mxINT64_CLASS:
        case mxUINT64_CLASS:
            szelemclass = 8;
            break;
        case mxSINGLE_CLASS:
        case mxINT32_CLASS:
        case mxUINT32_CLASS:
            szelemclass = 4;
            break;
        case mxCHAR_CLASS:
        case mxINT16_CLASS:
        case mxUINT16_CLASS:
            szelemclass = 2;
            break;
        case mxLOGICAL_CLASS:
        case mxINT8_CLASS:
        case mxUINT8_CLASS:
            szelemclass = 1;
            break;
        default:
            mexErrMsgTxt("INPLACEARRAY: Class not supported.");
    }
        
    if (mxIsSparse(AMAT)) /* check if sparse matrix */
        mexErrMsgTxt("INPLACEARRAY: sparse matrix not supported.");
    
    /* Offset */
    if (nrhs>=2) {
        if (!mxIsNumeric(OFFSET) || (mxGetClassID(OFFSET) != mxDOUBLE_CLASS))
            mexErrMsgTxt("INPLACEARRAY: Offset must be double numeric.");
        
        /* Empty offset array */
        if (mxGetM(OFFSET)==0 || mxGetN(OFFSET)==0)
            offset = 0;
        else if (mxGetM(OFFSET)!=1 || mxGetN(OFFSET)!=1)
            mexErrMsgTxt("INPLACEARRAY: Offset must be a scalar.");
        else
            /* Get the value */
            offset = (mwIndex)(*mxGetPr(OFFSET));
    }
    else offset = 0; /* No offset is provided */
    
    /* Number of dimensions */
    freedims = 0; /* default value */
    if (nrhs>=3) {
        if (!mxIsNumeric(SZ) || (mxGetClassID(SZ) != mxDOUBLE_CLASS))
            mexErrMsgTxt("INPLACEARRAY: SZ must be double numeric.");        
   
        /* Empty SZ array */
        if (mxGetM(SZ)==0 || mxGetN(SZ)==0)
        {
            nd = mxGetNumberOfDimensions(AMAT);
            dims = (mwSize*)mxGetDimensions(AMAT);
        }
        else {
            if (nrhs==3) /* Calling INPLACEARRAY(..., [N1 N2 ... Nn]) */
            {
                M = mxGetM(SZ);
                N = mxGetN(SZ);
                dblPr = mxGetPr(SZ);
                nd = M*N;
                if (nd<2) nd=2;
                dims = mxMalloc(nd*sizeof(mwSize)); freedims = 1;
                for (i=0; i<M*N; i++) dims[i] = (mwIndex)dblPr[i];
                for (i=M*N; i<nd; i++) dims[i] = 1;
                
            }    
            else /* Calling INPLACEARRAY(..., N1, N2, ..., Nn) */
            {
                M = nrhs-2;
                nd = M;
                if (nd<2) nd=2;
                dims = mxMalloc(nd*sizeof(mwSize)); freedims = 1;
                for (i=0; i<M; i++) dims[i] = (mwIndex)(*mxGetPr(prhs[2+i]));
                for (i=M; i<nd; i++) dims[i] = 1;
            }
        }
    }
    else /* No SZ is provided */
    {
        nd = mxGetNumberOfDimensions(AMAT);
        dims = (mwSize*)mxGetDimensions(AMAT);
    }
        
    /* Create the Matrix result (first output) */
    
    Pr = (char*)mxGetPr(AMAT);
    Pi = (char*)mxGetPi(AMAT);
    
    /* Check if the input is complex or not */
    ComplexFlag = ((Pi==NULL)? mxREAL:mxCOMPLEX);
    
    BMAT = mxCreateNumericMatrix(0, 0, ClassID, ComplexFlag);
    mxFree(mxGetPr(BMAT)); /* Free the data, normally Pr is NULL and this does
                              * nothing */

    /* Set the dimension as one column */
    flag = mxSetDimensions(BMAT, dims, nd);
    if (flag) /* Check for error */
        mexErrMsgTxt("INPLACEARRAY: cannot reshape the array as specified."); 
    
    /* Inplace data pointer of A */
    /* Equivalent to doing this: mxSetPr(BMAT, Pr + offset); 
       but access directly to data pointer in order to by pass Matlab
       checking, Thanks to James Tursa */
    offset *= szelemclass;
    ((Internal_mxArray*)(BMAT))->data.number_array.pdata = (Pr + offset);
    if (ComplexFlag)
    ((Internal_mxArray*)(BMAT))->data.number_array.pimag_data = (Pi + offset);
    
    /* Free the array of dimensions */
    if (freedims) mxFree(dims);
    
    return;
    
} /* Gateway of INPLACEARRAY.c */

