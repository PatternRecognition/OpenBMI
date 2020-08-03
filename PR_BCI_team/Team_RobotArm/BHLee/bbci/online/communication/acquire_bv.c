/*
  acquire_bv.c

  This file defines a mex-Function to communicate with the brainvision
  server via matlab. The commands are:
  
  state = acquire_bv(sampling_freq, server); 
  or
  state = acquire_bv(sampling_freq, server, sampleFilter);
  or    
  state = acquire_bv(sampling_freq, server, bFilter, aFilter);
  or    
  state = acquire_bv(sampling_freq, server, sampleFilter, bFilter, aFilter);
  
  
  The first call will use no filtering.
  The second one will use resampling filter.
  The third one will use IIR filter.
  The fourth one will use both.
 
  [data, blockno, marker_pos, marker_token, marker_description]
        = acquire_bv(state);
  acquire_bv('close');

  to initialize the connection, retrieve data, and close the
  connection respectively.
 
  NOTE: We observed a data loss when acquire_bv is called
        after a long period of time.
        To reproduce the error type:
          (first set your amplifier to produce sine waves)
 
         
          status = acquire_bv(1000,'localhost');
          dat1 = acquire_bv(status); pause(0.1); dat2 = acquire_bv(status);
          plot([dat1(end-120:end,1);
 
        You will notice a jump in the data.
 
        The jump occours when acquire_bv was idle for
        9 seconds. The problem lies in the brain vision recorder. 
        We did not tell or confirmed it from the company.
 
                                            Max Sagebaum 2008/03/17       
 

  This file has been initially written by Benjamin Blankertz, then
  edited by Guido Dornhege, and then even more
  heavily edited and extended by Mikio Braun, mikio@first.fhg.de
 
 
 
 2008/01/29 - Max Sagebaum 
                - add: check in the connection case if a connection exists
                - bug: the message data was not freed in abv_init
                - add: on packet failure the function will close the
                       connection or try to reconnect to the server
                       see acquire_bv.m for documentation
 2008/03/13 - Max Sagebaum
                - the lost of a packages was not calculated correctly
                - removed the check for a lost package
 
 2008/03/17 - Max Sagebaum
                - added IIR Filtering for the incoming data
 2008/07/01 - Max Sagebaum
                - a bug within the reconnect code was removed
                - the FIR (subsample) filter is now added to the out_state
                  and can be set by the user
 2008/09/18 - Max Sagebaum
                - If the FIR filter was not specified, it used a wrong
                  matlab structure.
                - I made some changes to remove the warning messages.
 2008/01/09 - Max Sagebaum
                - I moved the filters to an external file "filer.c" and 
                  "filter.h".
                - The programm writes the blocknumber only if the number of
                  return values is greater or equal to 2
 2009/09/18 - Max Sagebaum 
                - fixed a possible bug: If the channel selection was changed 
                    you got an invalid memory access exception
 2010/02/02 - Max Sagebaum
                - fixed a bug which showed in Season11. If you reconnected 
                    with acquire_bv in a second Matlab which was started from
                    a first one. You got a segmentation fault.
                    Removed the recursion in the getData method.
 2010/09/09 - Max Sagebaum
                - changed mxAssert into abv_assert
 2010/10/24 - Marton Danoczy
                - implemented receiving of int_32 data from Nouzz cap (if nType = 4)

 
 
  
  (c) Fraunhofer FIRST.IDA 2005
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

#ifdef _MSC_VER
#include "../../toolbox/fileio/msvc_stdint.h"
#else
#include <stdint.h>
#endif


#include "mex.h"
#include "brainserver.h"

#include "../winunix/winthreads.h"

#include "filter.h"

/*
 * DEFINES
 */

/* input and output arguments for the different uses of acquire_bv */

/* state = acquire_bv(sampling_freq, server) */
#define IN_FS        prhs[0]
#define IN_HOSTNAME  prhs[1]
#define OUT_STATE    plhs[0]
/* state = acquire_bv(sampling_freq, server, bFilter, aFilter) */
/*#define IN_B_FILTER  prhs[2]  */
/*#define IN_A_FILTER  prhs[3]  */

/* [data, blockno, marker_pos, marker_token, marker_desc ] 
   = acquire_bv(state); */
#define IN_STATE      prhs[0]
#define OUT_DATA      plhs[0]
#define OUT_BLOCK_NO  plhs[1]
#define OUT_MRK_POS   plhs[2]
#define OUT_MRK_TOE   plhs[3]
#define OUT_MRK_DESC  plhs[4]


#define MAX_CHARS 1024 /* maximum size of hostname */

/*
 * GLOBAL DATA
 */

const char *field_names[] = {
  "clab", "lag", "orig_fs", "scale", "block_no","chan_sel","reconnect","hostname","fir_filter"
};

#define NUMBER_OF_FIELDS (sizeof(field_names)/sizeof(*field_names))

static int connected = 0;

/*
 * FORWARD DECLARATIONS
 */ 

static void 
abv_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static void 
abv_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static double abv_filterDataIIR(double value, int channel) ;

static void 
abv_close();

static void abv_assert(bool condition,const char *text);

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
  /* Check arguments and set mode and parameters */
  if (nrhs == 1 && mxIsStruct(IN_STATE)) {
    abv_assert(nlhs < 6, "five output arguments maximum");
    abv_assert(nrhs == 1, "exactly one input argument required");
    abv_assert(mxIsStruct(IN_STATE), "input argument must be struct");
    if (!connected)
      mexWarnMsgTxt("acquire_bv: open a connection first!");
    else
      abv_getdata(nlhs, plhs, nrhs, prhs);
  }
  else if ((nrhs == 0) || (nrhs == 1 && mxIsChar(IN_STATE))) {
    abv_assert(nlhs == 0, "no output argument expected");
    if (!connected)
      mexWarnMsgTxt("acquire_bv: open a connection first!");
    else
      abv_close(nlhs, plhs, nrhs, prhs);
  }
  else {
    bool twoArgs = nrhs == 2
     && mxIsNumeric(prhs[0])
     && mxIsChar(prhs[1]);
    bool threeArgs = nrhs == 3
     && mxIsNumeric(prhs[0])
     && mxIsChar(prhs[1])
     && mxIsNumeric(prhs[2]);
    bool fourArgs = nrhs == 4
     && mxIsNumeric(prhs[0])
     && mxIsChar(prhs[1])
     && mxIsNumeric(prhs[2])
     && mxIsNumeric(prhs[3]);
    bool fiveArgs = nrhs == 5
     && mxIsNumeric(prhs[0])
     && mxIsChar(prhs[1])
     && mxIsNumeric(prhs[2])
     && mxIsNumeric(prhs[3])
     && mxIsNumeric(prhs[4]);
    if(twoArgs || threeArgs || fourArgs || fiveArgs) { 
      if (connected) {
        mexWarnMsgTxt("acquire_bv: connection is still open!");
        mexWarnMsgTxt("acquire_bv: closing the connection!");

        abv_close();
      }

      abv_init(nlhs, plhs, nrhs, prhs);
    } else { 
      mexWarnMsgTxt("acquire_bv: called with illegal parameters\n");
    }
  }
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
  char *bv_hostname = 0, *default_hostname= "brainamp";
  struct RDA_MessageStart *pMsgStart;
  
  /* Check input arguments */
  abv_assert(nrhs <= 5, "Four input arguments maximum.");
  abv_assert(mxIsNumeric(IN_FS), "First argument must be real scalar.");
  abv_assert(mxGetM(IN_FS) * mxGetN(IN_FS) == 1, 
	   "First argument must be real scalar.");

  /* Get server name (or use default "brainamp") */
  if (nrhs >= 2) {
    abv_assert(mxIsChar(IN_HOSTNAME), "Second argument must be a string.");
    bv_hostname = (char *) malloc(MAX_CHARS);
    mxGetString(IN_HOSTNAME, bv_hostname, MAX_CHARS);
  }
  else {
    bv_hostname = (char*) malloc(strlen(default_hostname) + 1);
    strcpy(bv_hostname, default_hostname);
  }

  /* open connection */
  result = initConnection(bv_hostname,&pMsgStart);
    
  if (result != IC_OKAY) {
    /* on error just return an empty matrix */
    OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
 }
  else {
    /* construct connection state structure */
    int nChans, lag, n;
    double orig_fs;
    mxArray *pArray;
    char *pChannelName;
    double *chan_sel;
    int i;
    const mxArray *IN_B_FILTER;
    const mxArray *IN_A_FILTER;
    const mxArray *RESAMPLE_FILTER;
    
    bool iirFilterSet;
    bool resampleFilterSet;
    
    
    nChans = pMsgStart->nChannels;
    orig_fs = 1000000.0 / ((double) pMsgStart->dSamplingInterval);
    lag = (int) (orig_fs / mxGetScalar(IN_FS));
    
    abv_assert(lag * (int)mxGetScalar(IN_FS) == (int)orig_fs,"The base frequency has to be a multiple of the requested frequency.");
    OUT_STATE = mxCreateStructMatrix(1, 1, NUMBER_OF_FIELDS, field_names);

    /* some simple labels */
    mxSetField(OUT_STATE, 0, "orig_fs", mxCreateDoubleScalar(orig_fs));
    mxSetField(OUT_STATE, 0, "lag", mxCreateDoubleScalar(lag));
    mxSetField(OUT_STATE, 0, "block_no", mxCreateDoubleScalar(-1.0));
    mxSetField(OUT_STATE, 0, "reconnect", mxCreateDoubleScalar(0));
    mxSetField(OUT_STATE, 0, "hostname", mxCreateString(bv_hostname));
    
    /* channel labels */
    pArray = mxCreateCellMatrix(1, nChans);    
    /* this odd hack is because pMsgStart contains several variably
       sized arrays, and this is the way to get the channel names 
     */ 
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
    for (n = 0;n<nChans;n++) {
        chan_sel[n] = n+1;
    }
    memcpy(mxGetPr(pArray), chan_sel, nChans*sizeof(double));
    free(chan_sel);
    mxSetField(OUT_STATE, 0, "chan_sel", pArray);
    
    /* check which type of filter to create */
    if(nrhs == 2) {
      /* no filter was set creating default */
      iirFilterSet = false;
      resampleFilterSet = false;
      
    } else if(nrhs == 3) {
      /* only a resampling filter */
      iirFilterSet = false;
      resampleFilterSet = true;
      RESAMPLE_FILTER = prhs[2];
    } else if(nrhs == 4) {
      /* only IIR filter */
      iirFilterSet = true;
      resampleFilterSet = false;
       IN_B_FILTER = prhs[2];
       IN_A_FILTER = prhs[3];
    } else if(nrhs == 5) {
      /* resample and IIR */
        iirFilterSet = true;
        resampleFilterSet = true;
        RESAMPLE_FILTER = prhs[2];
        IN_B_FILTER = prhs[3];
        IN_A_FILTER = prhs[4];
    }
    
     /* create the resample filter buffers */
    if(resampleFilterSet) {      
      double* filter;
      
      /* check the dimensions */
      if(mxGetM(RESAMPLE_FILTER) != 1) {
        mexErrMsgTxt("resample filter has to be a vector.");
        abv_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      if(mxGetN(RESAMPLE_FILTER) != lag) {
        mexErrMsgTxt(" resample filter has to correspondent with the sampling rate.");
        abv_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      
      /* we have to copy the matlab structure into a new one other wise the use of a own fir 
       * filter will produce a matlab error */
      filter = mxGetPr(RESAMPLE_FILTER);
      RESAMPLE_FILTER = mxCreateDoubleMatrix(1, lag, mxREAL); 
      memcpy(mxGetPr(RESAMPLE_FILTER),filter,lag * sizeof(double));
      filter = mxGetPr(RESAMPLE_FILTER);

      filterFIRCreate(filter, lag,nChans);
    } else {
      double* filter;
      
      /* the defalut filter will only take the last value from each block  */
      RESAMPLE_FILTER = mxCreateDoubleMatrix(1, lag, mxREAL); 
      filter = mxGetPr(RESAMPLE_FILTER);


      for(i = 0; i < lag;++i) {
        filter[i] = 0.0;
      }
      filter[lag - 1] = 1.0;

      filterFIRCreate(filter,lag, nChans);
    }
    mxSetField(OUT_STATE, 0, "fir_filter", (mxArray *)RESAMPLE_FILTER);
        
    /* create the filter buffers */
    if(iirFilterSet) {
      /* check the dimensions  */
      if(mxGetM(IN_B_FILTER) * mxGetM(IN_A_FILTER) != 1) {
        mexErrMsgTxt(" bFilter and aFilter has to be a vector.");
        abv_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      if(mxGetN(IN_B_FILTER) != mxGetN(IN_A_FILTER)) {
        mexErrMsgTxt(" bFilter and aFilter must have the same size.");
        abv_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }

      filterIIRCreate(mxGetPr(IN_A_FILTER), mxGetPr(IN_B_FILTER), mxGetN(IN_B_FILTER), nChans);
    } else {
      filterIIRCreate(NULL,NULL,1,nChans); /* create the default IIR filter */   
    }
    
    /* aaand, we're done */
    connected = 1;
  }
  
  free(bv_hostname);
  free(pMsgStart);
}

/************************************************************
 *
 * Get Data from the server
 *
 ************************************************************/

static void 
abv_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  int result;       /* return values for called function */
  mxArray *pArray;  /* generic pointer to a matlab array */
  int lastBlock;    /* the last block which we have seen */
  int nChannels;    /* number of channels */
  int blocksize;    /* the size of blocks */
  int ElementSize;  /* size of one data point. 2 if int16, 4 if int32, etc */
  int reconnect;    /* if we reconnect on connection loss */
  char *bv_hostname = 0;  /* if we reconnect wi need the hostname */
  struct RDA_MessageData *pMsgData;
  struct RDA_MessageStart *pMsgStart;
  /* get the information from the state and obtain the data */
  pArray = mxGetField(IN_STATE, 0, "block_no");
  lastBlock = (int) mxGetScalar(pArray);
  
  pArray= mxGetField(IN_STATE, 0, "clab");
  nChannels = mxGetN(pArray);
  
  result = getData(&pMsgData, &blocksize, lastBlock, nChannels, &ElementSize);
  
  /* If everything is okay, construct the appropriate output. Else
   * return empty arrays.
   */
  if (result != -1) {
    
    int n;
    int nChans_orig, lag, nPoints, nChans_sel, nMarkers, pDstPosition;
    double *chan_sel, *scale, *pDst0, *pMrkPos, *pSrcDouble;
    struct RDA_Marker *pMarker;
    char *pszType, *pszDesc;
    
    /* check for missing blocks */
    /* update by Max Sagebaum 
     * the block was not set in the version I had.
     * I changed the logic to do it the right way.
     * But it produced errors in online experiments. 
     */
    /*int lost = pMsgData->nBlock - (lastBlock + pMsgData->nPoints / blocksize);
    if (lastBlock != -1 && lost != 0 ) {
      mexPrintf("%d block(s) missed :", lost);
      mexWarnMsgTxt("");
   
    
    mxSetField(IN_STATE, 0, "block_no", mxCreateDoubleScalar(pMsgData->nBlock));
     */

    /* get necessary information from the current state */
    pArray = mxGetField(IN_STATE, 0, "clab");
    nChans_orig = mxGetN(pArray);

    pArray = mxGetField(IN_STATE, 0, "lag");
    lag = (int) mxGetScalar(pArray);

    nPoints = (getFIRPos() + pMsgData->nPoints)/lag;

    pArray = mxGetField(IN_STATE, 0, "chan_sel");
    chan_sel = mxGetPr(pArray);
    nChans_sel = mxGetN(pArray);
    
    /* check for the new resample filter */
    pArray = mxGetField(IN_STATE, 0, "fir_filter");
    if(mxGetM(pArray) != 1) {
      mexErrMsgTxt("new resample filter has to be a vector.");
      abv_close();
      OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
      return;
    }
    if(mxGetN(pArray) != lag) {
      mexErrMsgTxt("new resample filter has to correspondent with the sampling rate.");
      abv_close();
      OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
      return;
    }
    filterFIRSet(mxGetPr(pArray));
    
    /* the output block number */
    if(nlhs >= 2) {
      OUT_BLOCK_NO =
        mxCreateDoubleScalar((double)pMsgData->nBlock*blocksize/lag);
    }
    
    /* construct the data output matrix. */
    OUT_DATA = mxCreateDoubleMatrix(nPoints, nChans_sel, mxREAL);
    
    pArray = mxGetField(IN_STATE, 0, "scale");
    scale = mxGetPr(pArray);
    pDst0= mxGetPr(OUT_DATA);
    pDstPosition = 0;

    /* convert the source data to double format */
    pSrcDouble = malloc(pMsgData->nPoints * nChans_orig * sizeof(double));
    if (ElementSize==2) {
        int16_t *pSrc = pMsgData->nData;
        for(n = 0; n != pMsgData->nPoints * nChans_orig; ++n)
            pSrcDouble[n] = (double)pSrc[n];
    } else if (ElementSize==4) {
        int32_t *pSrc = (int32_t*) pMsgData->nData;
        for(n = 0; n != pMsgData->nPoints * nChans_orig; ++n)
            pSrcDouble[n] = (double)pSrc[n];
    } else
        mexErrMsgTxt("Unknown element size");
    
    /* filter the data with the filters */
    filterData(pSrcDouble,pMsgData->nPoints ,pDst0,nPoints, chan_sel, nChans_sel, scale);
    free(pSrcDouble);
    
    /* if markers are also requested, construct the appropriate output
       matrices */ 
    if (nlhs >= 3) {
      nMarkers = pMsgData->nMarkers;

      if (nMarkers > 0) {
        /* if markers existed, collect them */
        OUT_MRK_POS = mxCreateDoubleMatrix(1, nMarkers, mxREAL);
        pMrkPos = mxGetPr(OUT_MRK_POS);

        if (nlhs >= 4) {
      /*	  OUT_MRK_TOE = mxCreateDoubleMatrix(1, nMarkers, mxREAL);*/
          OUT_MRK_TOE = mxCreateCellMatrix(1,nMarkers);
      /*	  pMrkToe = mxGetPr(OUT_MRK_TOE);*/
        }

        if (nlhs == 5) 
          OUT_MRK_DESC = mxCreateCellMatrix(1, nMarkers);

        pMarker = (struct RDA_Marker*)
          ((char*)pMsgData->nData + 
           pMsgData->nPoints * nChans_orig * ElementSize);

        for (n = 0; n < nMarkers; n++) {
          pMrkPos[n]= pMarker->nPosition/lag;
          pszType = pMarker->sTypeDesc;
          pszDesc = pszType + strlen(pszType) + 1;
          if (nlhs >= 4)
            mxSetCell(OUT_MRK_TOE, n, mxCreateString(pszDesc));
      /*	    pMrkToe[n]= ((*pszDesc =='R') ? -1 : 1) * atoi(pszDesc+1); */
          if (nlhs == 5)
            mxSetCell(OUT_MRK_DESC, n, mxCreateString(pszType));

          pMarker = (struct RDA_Marker*)((char*)pMarker + pMarker->nSize);
        }

      }
      else {
        /* no markers -> return empty matrix */
        OUT_MRK_POS = mxCreateDoubleMatrix(0, 0, mxREAL);
        if (nlhs >= 4) {OUT_MRK_TOE = mxCreateDoubleMatrix(0, 0, mxREAL);};
        if (nlhs == 5) {OUT_MRK_DESC = mxCreateCellMatrix(0, 0);};
      }
    } /* end constructing marker outputs */
  }
  else {
    int nChans_sel;
    
    pArray = mxGetField(IN_STATE, 0, "reconnect");
    reconnect = (int) mxGetScalar(pArray);
    if(1 == reconnect) {
      printf("getData didn't work, reconnecting ");

      /* only close the connection */
      closeConnection();
      connected = 0;
      
      bv_hostname = (char *) malloc(MAX_CHARS);
      /* getting the hostname for the new connection */
      pArray = mxGetField(IN_STATE, 0, "hostname");
      mxGetString(pArray, bv_hostname, MAX_CHARS);
      
      /* try reconnecting till we get a new connection */
      while(IC_OKAY != (result = initConnection(bv_hostname, &pMsgStart))){
        printf("connecting failed, trying again\n");
      }
      
      /* cleaning things up */
      free(bv_hostname);
      free(pMsgStart);
      connected = 1;
    } else {
      printf("getData didn't work, closing connection, returning -2\n ");
      /* close the connection and clean everything up */
      abv_close();
    }
    
    /* We have an error in the data transmition return an empty datablock. */
    pArray = mxGetField(IN_STATE, 0, "chan_sel");
    nChans_sel = mxGetN(pArray);

    OUT_BLOCK_NO = mxCreateDoubleScalar(-2);
    OUT_DATA = mxCreateDoubleMatrix(0, nChans_sel, mxREAL);

    if (nlhs >= 3){OUT_MRK_POS = mxCreateDoubleMatrix(0,0, mxREAL);};
    if (nlhs >= 4){OUT_MRK_TOE = mxCreateDoubleMatrix(0,0, mxREAL);};
    if (nlhs == 5){OUT_MRK_DESC = mxCreateCellMatrix(0, 0);};
  }
  free(pMsgData);

}

/************************************************************
 *
 * Initialize Connection
 *
 ************************************************************/

static void abv_close()
{
  closeConnection();
  connected = 0;
  
  filterClose();
}

/************************************************************
 *
 * checks for errors and does some cleanup before returning to matlab
 * INPUT: condition       - The condition for the assert. Everything is ok
 *                          if condition is equal to one.
 *        text            - The text which is printed when condition ist not equal to one
 *
 ************************************************************/
static void abv_assert(bool condition,const char *text) {
  if(0 == condition) {
    abv_close();
    
    mexErrMsgTxt(text);
  }
}



/* 
   This is a little test program.

   I haven't yet figured out if it possible to compile against the
   matlab libraries without using mex. So for now, let's exclude this
   file here.
*/

#ifdef __NEVER__
int main(int argc, char **argv)
{
    const char *bv_hostname = "brainamp";
    struct RDA_MessageStart *pMsgStart;
    /*    struct RDA_MessageHeader *header; */
    struct RDA_MessageData *pMsgData;
    int numele = 0;
    int nChannels; 
    int blocksize;
    initConnection(bv_hostname, &pMsgStart);
    nChannels = pMsgStart->nChannels;
    free(pMsgStart);
    Sleep(1000);
    getData(&pMsgData, &blocksize, -1, nChannels);
    free(pMsgData);
    Sleep(100);
    closeConnection();
    printf("closed\n");
    /* empty queue */
    Sleep(5000);
    return 0;
}
#endif
