/*
  acquire_bv_braunschweig.c

  This file defines a mex-Function to communicate with the DAQ
  server via matlab. The commands are:
  
  state = acquire_bv_braunschweig(sampling_freq, server);
  [data, blockno, marker_pos, marker_token, marker_description]
        = acquire_bv_braunschweig(state);
  acquire_bv_braunschweig('close');

  to initialize the connection, retrieve data, and close the
  connection respectively.

  Compile in Matlab under Windows with
    mex acquire_bv_braunschweig.c brainserver.c headerfifo.c ws2_32.lib
    
  This file has been initially written by Guido Dornhege (?), but has
  been heavily edited and extended by Mikio Braun, mikio@first.fhg.de.
  Modified for the use with the DAQ server by Benjamin.
  
  (c) Fraunhofer FIRST.IDA 2006
*/

/*
 * This file only contains the driver functions for the communication
 * with matlab. The communication with the server can be found in
 * "brainserver.h".
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <ctype.h>

#include "mex.h"
#include "brainserver_braunschweig.h"

#include "../bbci_bet_unstable/winunix/winthreads.h"

/*
 * DEFINES
 */

/* input and output arguments for the different uses of acquire_bv */

/* state = acquire_bv_braunschweig(sampling_freq, server) */
#define IN_FS        prhs[0]
#define IN_HOSTNAME  prhs[1]
#define OUT_STATE    plhs[0]

/* [data, blockno, marker_pos, marker_token, marker_desc ] 
   = acquire_bv_braunschweig(state); */
#define IN_STATE      prhs[0]
#define OUT_DATA      plhs[0]
#define OUT_BLOCK_NO  plhs[1]


#define MAX_CHARS 1024 /* maximum size of hostname */

/*
 * GLOBAL DATA
 */

const char *field_names[] = {
  "clab", "lag", "orig_fs", "scale", "block_no","chan_sel"
};

#define NUMBER_OF_FIELDS (sizeof(field_names)/sizeof(*field_names))

static int connected = 0;

/*
 * FORWARD DECLARATIONS
 */ 

static void 
abv_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static int 
abv_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static int 
abv_close(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

#ifdef AC_THREADED

/* thread control */
/*int startPollThread();
int stopPollThread();
int beginFIFO();
int finishedFIFO();
void printThreadState();*/

#endif

/************************************************************
 *
 * mexFunction
 *
 ************************************************************/

/* decide in which of the three modes we are, and do some
 * input argument checking.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  enum ModeType {INIT, DATA, EXIT} mode;
  int result;
  mxArray *pArray;
  char *bv_hostname, *default_hostname= "brainamp";
  int lastBlock;
  int nChannels;
  
  /* Check arguments and set mode and parameters */
  if (nrhs == 1 && mxIsStruct(IN_STATE)) {
    mxAssert(nlhs > 4, "four output arguments maximum");
    mxAssert(nrhs != 1, "exactly one input argument required");
    mxAssert(!mxIsStruct(IN_STATE), "input argument must be struct");
    if (!connected)
      mexWarnMsgTxt("acquire_bv_braunschweig: open a connection first!");
    else
      abv_getdata(nlhs, plhs, nrhs, prhs);
  }
  else if ((nrhs == 0) || (nrhs == 1 && mxIsChar(IN_STATE))) {
    mxAssert(nlhs == 0, "no output argument expected");
    if (!connected)
      mexWarnMsgTxt("acquire_bv_braunschweig: open a connection first!");
    else
      abv_close(nlhs, plhs, nrhs, prhs);
  }
  else if (nrhs == 2 
	   && mxIsNumeric(prhs[0])
	   && mxIsChar(prhs[1]))
    abv_init(nlhs, plhs, nrhs, prhs);
  else 
    mexWarnMsgTxt("acquire_bv_braunschweig: called with illegal parameters\n");
}

/************************************************************
 *
 * Initialize Connection
 *
 ************************************************************/

static void 
abv_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int result;
  mxArray *pArray;
  char *bv_hostname = 0, *default_hostname= "brainamp";
  int lastBlock;
  int nChannels;
  struct RDA_MessageStart *pMsgStart;
  
  /* Check input arguments */
  mxAssert(nrhs <= 2, "Two input arguments maximum.");
  mxAssert(mxIsNumeric(IN_FS), "First argument must be real scalar.");
  mxAssert(mxGetM(IN_FS) * mxGetN(IN_FS) == 1, 
	   "First argument must be real scalar.");

  /* Get server name (or use default "brainamp") */
  if (nrhs == 2) {
    mxAssert(mxIsChar(IN_HOSTNAME), "Second argument must be a string.");
    bv_hostname = (char *) malloc(MAX_CHARS);
    mxGetString(IN_HOSTNAME, bv_hostname, MAX_CHARS);
  }
  else {
    bv_hostname = (char*) malloc(strlen(default_hostname) + 1);
    strcpy(bv_hostname, default_hostname);
  }

  /* open connection */
  result = initConnection(bv_hostname, &pMsgStart);
  free(bv_hostname);
  
  if (result != IC_OKAY) {
    /* on error just return an empty matrix */
    OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
    free(pMsgStart);
    return;
  }
  else {
    /* construct connection state structure */
    int nChans, lag, n;
    double orig_fs;
    mxArray *pArray;
    char *pChannelName;
    double *chan_sel;
    
    nChans = pMsgStart->nChannels;
    orig_fs = 1000000.0 / ((double) pMsgStart->dSamplingInterval);
    lag = (int) (orig_fs / mxGetScalar(IN_FS));
    
    OUT_STATE = mxCreateStructMatrix(1, 1, NUMBER_OF_FIELDS, field_names);

    /* some simple labels */
    mxSetField(OUT_STATE, 0, "orig_fs", mxCreateDoubleScalar(orig_fs));
    mxSetField(OUT_STATE, 0, "lag", mxCreateDoubleScalar(lag));
    mxSetField(OUT_STATE, 0, "block_no", mxCreateDoubleScalar(-1.0));
    
    /* channel labels */
    pArray = mxCreateCellMatrix(1, nChans);    
    /* this odd hack is because pMsgStart contains several variably
       sized arrays, and this is the way to get the channel names 
                           |
                   vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv */ 
    pChannelName = (char *) ((double*) pMsgStart->dResolutions + nChans);
    for (n = 0; n < nChans; n++) {
      mxSetCell(pArray, n, mxCreateString(pChannelName));
      pChannelName += strlen(pChannelName) + 1;
    }
    mxSetField(OUT_STATE, 0, "clab", pArray);
    
    /* channel scale factors */
    pArray = mxCreateDoubleMatrix(1, nChans, mxREAL);
    memcpy(mxGetPr(pArray), pMsgStart->dResolutions, nChans*sizeof(double));
    mxSetField(OUT_STATE, 0, "scale", pArray);

    /* channel indices */
    pArray = mxCreateDoubleMatrix(1, nChans, mxREAL);
    chan_sel = (double *) malloc(nChans*sizeof(double));
    for (n = 0;n<nChans;n++)
      chan_sel[n] = n+1;
    memcpy(mxGetPr(pArray), chan_sel, nChans*sizeof(double));
    free(chan_sel);
    mxSetField(OUT_STATE, 0, "chan_sel", pArray);
    
    /* aaand, we're done */
    connected = 1;
  }
}

/************************************************************
 *
 * Get Data from the server
 *
 ************************************************************/

static int 
abv_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int result;       /* return values for called function */
  mxArray *pArray;  /* generic pointer to a matlab array */
  int lastBlock;    /* the last block which we have seen */
  int nChannels;    /* number of channels */
  int blocksize;    /* the size of blocks */
  struct RDA_MessageData32 *pMsgData;
  
  /* get the information from the state and obtain the data */
  pArray = mxGetField(IN_STATE, 0, "block_no");
  lastBlock = (int) mxGetScalar(pArray);
  
  pArray= mxGetField(IN_STATE, 0, "clab");
  nChannels = mxGetN(pArray);
  
  result = getData(&pMsgData, &blocksize, lastBlock, nChannels);
  /* If everything is okay, construct the appropriate output. Else
   * return empty arrays.
   */
  if (result != -1) {
    int t, n, c;
    int nChans_orig, lag, nPoints, nChans_sel;
    double *chan_sel, *scale, *pDst, *pDst0;
    float *pSrc;

    /*  if (pMsgData->fData) {
    mexPrintf("nonzero data received [3]: %f\n", pMsgData->fData);
    }*/
    /* check for missing blocks */
    if (lastBlock != -1 && pMsgData->nBlock > lastBlock + 1) {
      mexPrintf("%d block(s) missed :", pMsgData->nBlock-lastBlock-1);
      mexWarnMsgTxt("");
    }

    /* get necessary information from the current state */
    pArray = mxGetField(IN_STATE, 0, "clab");
    nChans_orig = mxGetN(pArray);

    pArray = mxGetField(IN_STATE, 0, "lag");
    lag = (int) mxGetScalar(pArray);

    nPoints = pMsgData->nPoints/lag;

    pArray = mxGetField(IN_STATE, 0, "chan_sel");
    chan_sel = mxGetPr(pArray);
    nChans_sel = mxGetN(pArray);

    /* the output block number */
    OUT_BLOCK_NO =
      mxCreateDoubleScalar((double)pMsgData->nBlock*blocksize/lag);

    /* construct the data output matrix. Copy the data (re-arranging
       the channels according to chan_sel, scaling the values
       according to scale) */
    OUT_DATA = mxCreateDoubleMatrix(nPoints, nChans_sel, mxREAL);      

    /*mexPrintf("OUT_DATA is an %d x %d matrix\n",
      mxGetN(OUT_DATA), mxGetM(OUT_DATA));*/

    pArray = mxGetField(IN_STATE, 0, "scale");
    scale = mxGetPr(pArray) - 1;
    pDst0= mxGetPr(OUT_DATA);
    pSrc = pMsgData->fData - 1;
 
    for (t = 0; t < nPoints; t++) {
      pDst = pDst0 + t;
      for (n = 0; n < nChans_sel; n++) {
	c = chan_sel[n];
	*pDst = (double)pSrc[c];
	pDst+= nPoints;
      }
      pSrc+= nChans_orig * lag;
    }
    
  }
  else {
    printf("getData didn't work, returning -2\n");
    OUT_BLOCK_NO = mxCreateDoubleScalar(-2);
  }
  free(pMsgData);

}

/************************************************************
 *
 * Initialize Connection
 *
 ************************************************************/

static int 
abv_close(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  closeConnection();
  connected = 0;
}
