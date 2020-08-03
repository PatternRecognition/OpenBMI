/*
  acquire_sigserv_filter.c

  This file defines a mex-Function to filter the data we get from the signal
  server.
  
  acquire_sigserv_filter('createFilter', firFilter, aFilt, bFilt, nChannels)
  acquire_sigserv_filter('setFIR', firFilter)
  acquire_sigserv_filter('delete')
  filData = acquire_sigserv_filter('filter',data, state)
  
  First you need to create the FIR or IIR filter.
  Then you can start filtering the data. Whenn you are finished you need to delete
  the filters.
  
 2010/08/26 - Max Sagebaum 
                - file created used acquire_bv as a base. 
  
  (c) Fraunhofer FIRST.IDA 2010
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <ctype.h>

#include "mex.h"
#include "filter.h"

/*
 * DEFINES
 */
#define NUM_INPUTS_CREATE_FILTER 5
#define NUM_OUTPUTS_CREATE_FILTER 0
#define POS_CREATE_FILTER_FIR    1
#define POS_CREATE_FILTER_IIR_A  2
#define POS_CREATE_FILTER_IIR_B  3
#define POS_CREATE_FILTER_CHANNELS   4

#define NUM_INPUTS_SET_FIR    2
#define NUM_OUTPUTS_SET_FIR    0
#define POS_SET_FIR_FILTER    1

#define NUM_INPUTS_DELETE 1
#define NUM_OUTPUTS_DELETE 0

#define NUM_INPUTS_FILTER 3
#define NUM_OUTPUTS_FILTER 1
#define POS_FILTER_DATA      1
#define POS_FILTER_STATE     2
#define POS_FILTER_FILTDATA  0

static int channelCount;              /* the number of bci channels */

/*
 * FORWARD DECLARATIONS
 */ 

static void 
assf_createFilter(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static void 
assf_setFIR(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static void 
assf_delete(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static void 
assf_filter(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

/************************************************************
 *
 * mexFunction
 *
 ************************************************************/

/* 
 * check the options and the number of arguments.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  char *optionString;
  int optionStringSize;
  if(nrhs == 0) {
    mexErrMsgTxt("acquire_sigserv_filter: At leas one argument required.");
  }
  if(!mxIsChar(prhs[0])) {
    mexErrMsgTxt("acquire_sigserv_filter: First argument has to be a string.");
  }
  
  /* Get the option string */
  optionStringSize = (mxGetM(prhs[0]) * mxGetN(prhs[0]) * sizeof(mxChar)) + 1;
  optionString = (char *) malloc(optionStringSize * sizeof(char));
  mxGetString(prhs[0], optionString, optionStringSize);
  
  if(strcmp (optionString,"createFilter") == 0) {
    if(nlhs != NUM_OUTPUTS_CREATE_FILTER || nrhs != NUM_INPUTS_CREATE_FILTER) {
       mexErrMsgTxt("acquire_sigserv_filter: Wrong number of arguments for createFilter.");
    }
    
    assf_createFilter(nlhs, plhs, nrhs, prhs);    
  } else if(strcmp (optionString,"setFIR") == 0) {
    if(nlhs != NUM_OUTPUTS_SET_FIR || nrhs != NUM_INPUTS_SET_FIR) {
       mexErrMsgTxt("acquire_sigserv_filter: Wrong number of arguments for setFIR.");
    }
    
    assf_setFIR(nlhs, plhs, nrhs, prhs);    
  } else if(strcmp (optionString,"delete") == 0) {
    if(nlhs != NUM_OUTPUTS_DELETE || nrhs != NUM_INPUTS_DELETE) {
       mexErrMsgTxt("acquire_sigserv_filter: Wrong number of arguments for delete.");
    }
    
    assf_delete(nlhs, plhs, nrhs, prhs);    
  } else if(strcmp (optionString,"filter") == 0) {
    if(nlhs != NUM_OUTPUTS_FILTER || nrhs != NUM_INPUTS_FILTER) {
       mexErrMsgTxt("acquire_sigserv_filter: Wrong number of arguments for filter.");
    }
    
    assf_filter(nlhs, plhs, nrhs, prhs);    
  } else {
    mexWarnMsgTxt("acquire_sigserv_filter: option not reconized!");
  }
  
  free(optionString);
}

/************************************************************
 *
 * Create the fir filter
 *
 ************************************************************/
static void
assf_createFilter(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  double* firFilter;
  int firFilterSize;
  double* iirFilterA;
  double* iirFilterB;
  int iirFilterSize;
  
  /* check the dimensions */
  if(mxGetM(prhs[POS_CREATE_FILTER_FIR]) != 1) {
    mexErrMsgTxt("acquire_sigserv_filter: resample filter has to be a vector.");
    return;
  }
  
  if(mxGetM(prhs[POS_CREATE_FILTER_CHANNELS]) != 1 || mxGetN(prhs[POS_CREATE_FILTER_CHANNELS]) != 1) {
    mexErrMsgTxt("acquire_sigserv_filter: number of channels has to be a scalar.");
    return;
  }
  
  if(mxGetM(prhs[POS_CREATE_FILTER_IIR_A]) != 1) {
    mexErrMsgTxt("acquire_sigserv_filter: iir filter a has to be a vector.");
    return;
  }
  
  if(mxGetM(prhs[POS_CREATE_FILTER_IIR_B]) != 1) {
    mexErrMsgTxt("acquire_sigserv_filter: iir filter b has to be a vector.");
    return;
  }
  
  if(mxGetN(prhs[POS_CREATE_FILTER_IIR_B]) != mxGetN(prhs[POS_CREATE_FILTER_IIR_A])) {
    mexErrMsgTxt("acquire_sigserv_filter: iir filter a and b must have the same size.");
    return;
  }

  /* create the fir filter */
  firFilter = mxGetPr(prhs[POS_CREATE_FILTER_FIR]);
  firFilterSize = (int)mxGetN(prhs[POS_CREATE_FILTER_FIR]);
  channelCount = (int)mxGetScalar(prhs[POS_CREATE_FILTER_CHANNELS]);      

  filterFIRCreate(firFilter, firFilterSize,channelCount);
  
  /* create the iir filter */
  iirFilterA = mxGetPr(prhs[POS_CREATE_FILTER_IIR_A]);
  iirFilterB = mxGetPr(prhs[POS_CREATE_FILTER_IIR_B]);
  iirFilterSize = (int)mxGetN(prhs[POS_CREATE_FILTER_IIR_A]);

  filterIIRCreate(iirFilterA, iirFilterB, iirFilterSize,channelCount);
}

/************************************************************
 *
 * Set the fir filter
 *
 ************************************************************/
static void 
assf_setFIR(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  double* firFilter;
  
  /* check the dimensions */
  if(mxGetM(prhs[POS_SET_FIR_FILTER]) != 1) {
    mexErrMsgTxt("acquire_sigserv_filter: new resample filter has to be a vector.");
    return;
  }
  
  if(mxGetN(prhs[POS_SET_FIR_FILTER]) != filterGetFIRSize()) {
    mexErrMsgTxt("acquire_sigserv_filter: new resample filter must have the same size.");
    return;
  }

  firFilter = mxGetPr(prhs[POS_SET_FIR_FILTER]);
  
  filterFIRSet(firFilter);
}

/************************************************************
 *
 * Delete the filter and the buffers.
 *
 ************************************************************/
static void 
assf_delete(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  filterClose();
}

/************************************************************
 *
 * Filter the data with the created filters.
 *
 ************************************************************/
static void 
assf_filter(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int lag;
  double *chan_sel, *scale, *pDst, *pSrc;
  int nChans_orig;
  int nChans_sel;
  int nPoints_out;
  int nPoints_in;
  mxArray *pArray;  /* generic pointer to a matlab array */
  
  /* check the dimensions */
  if(mxGetM(prhs[POS_FILTER_DATA]) != channelCount) {
    mexErrMsgTxt("acquire_sigserv_filter: The data has to be a row matrix with the number of channels");
    return;
  }
 
  /* get necessary information from the current state */
  pArray = mxGetField(prhs[POS_FILTER_STATE], 0, "lag");
  lag = (int) mxGetScalar(pArray);

  pArray = mxGetField(prhs[POS_FILTER_STATE], 0, "chan_sel");
  chan_sel = mxGetPr(pArray);
  nChans_sel = mxGetN(pArray);        

  pArray = mxGetField(prhs[POS_FILTER_STATE], 0, "scale");
  scale = mxGetPr(pArray);

  /* get the size of the data and calculate the size of the output */
  nChans_orig = mxGetM(prhs[POS_FILTER_DATA]);
  nPoints_in = mxGetN(prhs[POS_FILTER_DATA]);
  nPoints_out = (getFIRPos() + nPoints_in)/lag;

  /* construct the data output matrix. */
  plhs[POS_FILTER_FILTDATA] = mxCreateDoubleMatrix(nPoints_out, nChans_sel, mxREAL);

  /* get the pointers for the data */
  pDst= mxGetPr(plhs[POS_FILTER_FILTDATA]);
  pSrc = mxGetPr(prhs[POS_FILTER_DATA]);

  /* filter the data with the filters */
  filterData(pSrc,nPoints_in ,pDst,nPoints_out, chan_sel, nChans_sel, scale);
}