/*
  acquire_emotive.c

  This file defines a mex-Function to communicate with the emotive device
  via matlab. The commands are:
  
  state = acquire_emotive(sampling_freq, server); 
  or
  state = acquire_emotive(sampling_freq, server, sampleFilter);
  or    
  state = acquire_emotive(sampling_freq, server, bFilter, aFilter);
  or    
  state = acquire_emotive(sampling_freq, server, sampleFilter, bFilter, aFilter);
  
  
  The first call will use no filtering.
  The second one will use resampling filter.
  The third one will use IIR filter.
  The fourth one will use both.
 
  [data, blockno, marker_pos, marker_token, marker_description]
        = acquire_emotive(state);
  acquire_emotive('close');

  to initialize the connection, retrieve data, and close the
  connection respectively.
 
 
 
 2011/04/01 - Max Sagebaum 
                - copy from acquire_emotiv.c. Logic for emotiv added.

 (c) Fraunhofer FIRST.IDA 2011
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

#include "filter.h"

/* 
 * emotive includes
 */

#include "emotiv\EmoStateDLL.h"
#include "emotiv\edk.h"
#include "emotiv\edkErrorCode.h"

/*
 * thread inclueds
 */
#include <windows.h>
#include "..\..\toolbox\utils\parallelPort\ioPort.c"

/*
 * DEFINES
 */

/* input and output arguments for the different uses of acquire_emotiv */

/* state = acquire_emotiv(sampling_freq, server) */
#define IN_FS        prhs[0]
#define IN_HOSTNAME  prhs[1]
#define OUT_STATE    plhs[0]
/* state = acquire_emotiv(sampling_freq, server, bFilter, aFilter) */
/*#define IN_B_FILTER  prhs[2]  */
/*#define IN_A_FILTER  prhs[3]  */

/* [data, blockno, marker_pos, marker_token, marker_desc ] 
   = acquire_emotiv(state); */
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

const EE_DataChannel_t EMOTIVE_CHANNEL_LIST[] = {
		ED_COUNTER,
		ED_AF3, ED_F7, ED_F3, ED_FC5, ED_T7, 
		ED_P7, ED_O1, ED_O2, ED_P8, ED_T8, 
		ED_FC6, ED_F4, ED_F8, ED_AF4, ED_GYROX, ED_GYROY, ED_TIMESTAMP, 
		ED_FUNC_ID, ED_FUNC_VALUE, ED_MARKER, ED_SYNC_SIGNAL
	};
const int EMOTIVE_CHANNEL_LIST_SIZE = sizeof(EMOTIVE_CHANNEL_LIST)/sizeof(EE_DataChannel_t);

const char *EMOTIVE_CHANNEL_HEADER_LIST[] = {"COUNTER","AF3","F7","F3","FC5","T7","P7","O1","O2","P8", 
                      "T8","FC6","F4","F8","AF4","GYROX","GYROY","TIMESTAMP",   
                      "FUNC_ID","FUNC_VALUE","MARKER","SYNC_SIGNAL"};
                      
const int EMOTIVE_CHANNEL_HEADER_LIST_SIZE = sizeof(EMOTIVE_CHANNEL_HEADER_LIST)/sizeof(*EMOTIVE_CHANNEL_HEADER_LIST);
const int MARKER_CHANNEL = 20;

static int connected = 0;
static EmoEngineEventHandle eEvent;
static DataHandle hData = NULL;
static unsigned int userID = 0;
static double lastMarker = 0.0;
/*
 * FORWARD DECLARATIONS
 */ 

static void 
aemo_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static void 
aemo_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static double aemo_filterDataIIR(double value, int channel) ;

static void 
aemo_close();

static bool aemo_connect(char *hostname);

static void aemo_assert(bool condition,const char *text);

/* values for the paralelport thread */
static int keepThreadAlive = 0;
static int isThreadRunning = 0;
static HANDLE threadHandle = NULL;
static unsigned short parallelportAddr = 0;
static HINSTANCE hLib = NULL;

static DWORD WINAPI readParallelport(LPVOID lpParameter);


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
    aemo_assert(nlhs < 6, "five output arguments maximum");
    aemo_assert(nrhs == 1, "exactly one input argument required");
    aemo_assert(mxIsStruct(IN_STATE), "input argument must be struct");
    if (!connected)
      mexWarnMsgTxt("acquire_emotiv: open a connection first!");
    else
      aemo_getdata(nlhs, plhs, nrhs, prhs);
  }
  else if ((nrhs == 0) || (nrhs == 1 && mxIsChar(IN_STATE))) {
    aemo_assert(nlhs == 0, "no output argument expected");
    if (!connected)
      mexWarnMsgTxt("acquire_emotiv: open a connection first!");
    else
      aemo_close(nlhs, plhs, nrhs, prhs);
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
        mexWarnMsgTxt("acquire_emotiv: connection is still open!");
        mexWarnMsgTxt("acquire_emotiv: closing the connection!");

        aemo_close();
      }

      aemo_init(nlhs, plhs, nrhs, prhs);
    } else { 
      mexWarnMsgTxt("acquire_emotiv: called with illegal parameters\n");
    }
  }
}

/************************************************************
 * 
 * Connect to the Emotive Engine
 *
 ************************************************************/

static bool aemo_connect(char *hostname) {
  if(strcmp(hostname, "EmoEngine") == 0) {
    return EE_EngineConnect("Emotiv Systems-5") == EDK_OK;
  } else {
    return EE_EngineRemoteConnect(hostname, 3008,"Emotiv Systems-5") == EDK_OK;
  }  
}

/************************************************************
 *
 * Initialize Connection
 *
 ************************************************************/

static void 
aemo_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  bool result;
  bool emotiveReady;
  char *emoComp_hostname = 0, *default_hostname= "EmoEngine";
  int state = 0;
  unsigned int emotiveSamplingRate = 0;
  EE_Event_t eventType;

  
  /* Check input arguments */
  aemo_assert(nrhs <= 5, "Four input arguments maximum.");
  aemo_assert(mxIsNumeric(IN_FS), "First argument must be real scalar.");
  aemo_assert(mxGetM(IN_FS) * mxGetN(IN_FS) == 1, 
	   "First argument must be real scalar.");

  /* Get server name (or use default "EmoEngine") */
  if (nrhs >= 2) {
    aemo_assert(mxIsChar(IN_HOSTNAME), "Second argument must be a string.");
    emoComp_hostname = (char *) malloc(MAX_CHARS);
    mxGetString(IN_HOSTNAME, emoComp_hostname, MAX_CHARS);
  }
  else {
    emoComp_hostname = (char*) malloc(strlen(default_hostname) + 1);
    strcpy(emoComp_hostname, default_hostname);
  }

  /* open connection */
  result = aemo_connect(emoComp_hostname);
    
  if (result != true) {
    /* on error just return an empty matrix */
    OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
    
    return;
 } 
  
  /* init the emotive event */
  eEvent = EE_EmoEngineEventCreate();
  
  /* now try to connect to the emotive engine */
  emotiveReady = false;
  while (!emotiveReady) {
    state = EE_EngineGetNextEvent(eEvent);

    if (EDK_OK == state ) {
      unsigned int userID;
      
      eventType = EE_EmoEngineEventGetType(eEvent);
      EE_EmoEngineEventGetUserId(eEvent, &userID);

      /* Log the EmoState if it has been updated */
      if (eventType == EE_UserAdded) {
        EE_DataAcquisitionEnable(userID,true);
        /* create the data store for the data */
        hData = EE_DataCreate();
        EE_DataSetBufferSizeInSec(7.0);
        EE_DataUpdateHandle(userID, hData);

        state =  EE_DataGetSamplingRate  (userID, &emotiveSamplingRate);
        if(EDK_OK != state){
            printf("Error in retriving emotive Sample rate. Setting it to 128");
            emotiveSamplingRate = 128;
        }
        
        printf("Emotive Sample rate: %u", emotiveSamplingRate);
        emotiveReady = true;
      }
    } else if(EDK_UNKNOWN_ERROR == state) {
      OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
    
      return;
    }
  }
  if(!emotiveReady) {
    /* on error just return an empty matrix */
    OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
    
    return;
  } else {
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
     
    nChans = EMOTIVE_CHANNEL_LIST_SIZE;
    orig_fs = (double)emotiveSamplingRate;
    lag = (int) (orig_fs / mxGetScalar(IN_FS));
    
    aemo_assert(lag * (int)mxGetScalar(IN_FS) == (int)orig_fs,"The base frequency has to be a multiple of the requested frequency.");
    OUT_STATE = mxCreateStructMatrix(1, 1, NUMBER_OF_FIELDS, field_names);

    /* some simple labels */
    mxSetField(OUT_STATE, 0, "orig_fs", mxCreateDoubleScalar(orig_fs));
    mxSetField(OUT_STATE, 0, "lag", mxCreateDoubleScalar(lag));
    mxSetField(OUT_STATE, 0, "block_no", mxCreateDoubleScalar(-1.0));
    mxSetField(OUT_STATE, 0, "reconnect", mxCreateDoubleScalar(0));
    mxSetField(OUT_STATE, 0, "hostname", mxCreateString(emoComp_hostname));
    
    /* channel labels */
    pArray = mxCreateCellMatrix(1, nChans);    
    for (n = 0; n < nChans; n++) {
      mxSetCell(pArray, n, mxCreateString(EMOTIVE_CHANNEL_HEADER_LIST[n]));
    }
    mxSetField(OUT_STATE, 0, "clab", pArray);
    
    /* channel scale factors */
    pArray = mxCreateDoubleMatrix(1, nChans, mxREAL);
    chan_sel = (double *) malloc(nChans*sizeof(double));
    for (n = 0;n<nChans;n++) {
        chan_sel[n] = 1.0;
    }
    memcpy(mxGetPr(pArray), chan_sel, nChans*sizeof(double));
    free(chan_sel);
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
        aemo_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      if(mxGetN(RESAMPLE_FILTER) != lag) {
        mexErrMsgTxt(" resample filter has to correspondent with the sampling rate.");
        aemo_close();
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
        aemo_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      if(mxGetN(IN_B_FILTER) != mxGetN(IN_A_FILTER)) {
        mexErrMsgTxt(" bFilter and aFilter must have the same size.");
        aemo_close();
        OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }

      filterIIRCreate(mxGetPr(IN_A_FILTER), mxGetPr(IN_B_FILTER), mxGetN(IN_B_FILTER), nChans);
    } else {
      filterIIRCreate(NULL,NULL,1,nChans); /* create the default IIR filter */   
    }
    
    /* create the thread */
    {
      mxArray* pArray;
      /* create the parallelport library */
      hLib = createLibrary();
      
      /* get the port of the parllel port */
      pArray = mexGetVariable("global", "IO_ADDR");      
      aemo_assert(NULL != pArray,"acquire_emotiv: Could not get IO_ADDR from the global workspace.");
      aemo_assert(1 == mxIsDouble(pArray)&& 1 == mxGetM(pArray) && 1 == mxGetN(pArray),"acquire_emotiv: IO_ADDR is no scalar value.");
      parallelportAddr = (unsigned short)mxGetScalar(pArray);
      
      /* create the thread for the parallelport */
      threadHandle = CreateThread(NULL, 0,readParallelport,NULL,CREATE_SUSPENDED,
                               NULL);

      aemo_assert(NULL != threadHandle,"acquire_emotiv: could not create parallelportthread");
      
      keepThreadAlive = 1;
      isThreadRunning = 1;
      aemo_assert(-1 != ResumeThread(threadHandle),"acquire_emotiv: could not start parallelportthread");    
    }
    
    /* aaand, we're done */
    connected = 1;
  }
  
  free(emoComp_hostname);
}

/************************************************************
 *
 * Get Data from the server
 *
 ************************************************************/

static double* aemo_getDataFromDevice(unsigned int *nSamplesTakenReturn) {
  double *dataBuffer;
  double *data = NULL;
  int i;
  int sampleIdx;
  int state;
  unsigned int nSamplesTaken;
  
  /* update the event queue */
  state = EE_EngineGetNextEvent(eEvent);

  /* Update the data in the data structure */
  EE_DataUpdateHandle(0, hData);
  
  nSamplesTaken=0;
  EE_DataGetNumberOfSample(hData,&nSamplesTaken);

  dataBuffer = malloc(nSamplesTaken * sizeof(double));
  data = malloc(nSamplesTaken * EMOTIVE_CHANNEL_LIST_SIZE * sizeof(double));
  if (nSamplesTaken != 0) {
    /* iterate over the channels and set the values into the data structure */
    for (i = 0 ; i<EMOTIVE_CHANNEL_LIST_SIZE ; i++) {
      EE_DataGet(hData, EMOTIVE_CHANNEL_LIST[i], dataBuffer, nSamplesTaken);
      
      for (sampleIdx=0 ; sampleIdx<(int)nSamplesTaken ; ++ sampleIdx) {        
        data[sampleIdx * EMOTIVE_CHANNEL_LIST_SIZE + i] = dataBuffer[sampleIdx];
      }	
    }
  }
  
  *nSamplesTakenReturn = nSamplesTaken;
  return data;
}

/************************************************************
 *
 * This function extracts the markers from the marker channel.
 *
 ************************************************************/

static int aemo_findMarkers(int* markerPositions, double* markerValues, double* data, int maximumMarkerSize) {
    int markerCount;
    int curDataPosition;
    
    markerCount = 0;
    for(curDataPosition = 0; curDataPosition < maximumMarkerSize; ++curDataPosition) {
        double curMarker;
        
        curMarker = data[curDataPosition * EMOTIVE_CHANNEL_LIST_SIZE + MARKER_CHANNEL];
        if(lastMarker != curMarker) {
            if(0.0 != curMarker) {
                markerPositions[markerCount] = curDataPosition;
                markerValues[markerCount] = curMarker;
                
                ++markerCount;
            }
            
            lastMarker = curMarker;
        }
        
    }
    
    return markerCount;
    
}


static void 
aemo_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  int result;       /* return values for called function */
  mxArray *pArray;  /* generic pointer to a matlab array */
  int lastBlock;    /* the last block which we have seen */
  int nChannels;    /* number of channels */
  int blocksize;    /* the size of blocks */
  int ElementSize;  /* size of one data point. 2 if int16, 4 if int32, etc */
  int reconnect;    /* if we reconnect on connection loss */
  char *emoComp_hostname = 0;  /* if we reconnect wi need the hostname */
  double *pSrcDouble;
  unsigned int nSampleNumber;
  /* get the information from the state and obtain the data */
  pArray = mxGetField(IN_STATE, 0, "block_no");
  lastBlock = (int) mxGetScalar(pArray);
  
  pArray= mxGetField(IN_STATE, 0, "clab");
  nChannels = mxGetN(pArray);
  
  pSrcDouble = aemo_getDataFromDevice(&nSampleNumber);
  blocksize = 1;
  
  /* If everything is okay, construct the appropriate output. Else
   * return empty arrays.
   */
  if (NULL != pSrcDouble) {
    int n;
    int nChans_orig, lag, nPoints, nChans_sel, nMarkers, pDstPosition;
    double *chan_sel, *scale, *pDst0, *pMrkPos;
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

    nPoints = (getFIRPos() + nSampleNumber)/lag;

    pArray = mxGetField(IN_STATE, 0, "chan_sel");
    chan_sel = mxGetPr(pArray);
    nChans_sel = mxGetN(pArray);
    
    /* check for the new resample filter */
    pArray = mxGetField(IN_STATE, 0, "fir_filter");
    if(mxGetM(pArray) != 1) {
      mexErrMsgTxt("new resample filter has to be a vector.");
      aemo_close();
      OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
      return;
    }
    if(mxGetN(pArray) != lag) {
      mexErrMsgTxt("new resample filter has to correspondent with the sampling rate.");
      aemo_close();
      OUT_STATE = mxCreateDoubleMatrix(0, 0, mxREAL);
      return;
    }
    filterFIRSet(mxGetPr(pArray));
    
    /* the output block number */
    if(nlhs >= 2) {
      OUT_BLOCK_NO =
        mxCreateDoubleScalar(lastBlock + (double)nSampleNumber*blocksize/lag);
    }
    
    /* construct the data output matrix. */
    OUT_DATA = mxCreateDoubleMatrix(nPoints, nChans_sel, mxREAL);
    
    pArray = mxGetField(IN_STATE, 0, "scale");
    scale = mxGetPr(pArray);
    pDst0= mxGetPr(OUT_DATA);
    pDstPosition = 0;
    
    /* filter the data with the filters */
    filterData(pSrcDouble,nSampleNumber ,pDst0,nPoints, chan_sel, nChans_sel, scale);
    
    /* if markers are also requested, construct the appropriate output
       matrices */ 
    if (nlhs >= 3) {
        int* markerPositions;
        double* markerValues;
        
        markerPositions = malloc(nSampleNumber * sizeof(int));
        markerValues = malloc(nSampleNumber * sizeof(double));
        nMarkers = aemo_findMarkers(markerPositions, markerValues,pSrcDouble, nSampleNumber);

      if (nMarkers > 0) {
        /* if markers existed, collect them */
        OUT_MRK_POS = mxCreateDoubleMatrix(1, nMarkers, mxREAL);
        memcpy(mxGetPr(OUT_MRK_POS),markerValues,nMarkers * sizeof(double));
        
        if (nlhs >= 4) {
      /*	  OUT_MRK_TOE = mxCreateDoubleMatrix(1, nMarkers, mxREAL);*/
          OUT_MRK_TOE = mxCreateCellMatrix(1,nMarkers);
      /*	  pMrkToe = mxGetPr(OUT_MRK_TOE);*/
        }

        if (nlhs == 5) 
          OUT_MRK_DESC = mxCreateCellMatrix(1, nMarkers);
        
        for (n = 0; n < nMarkers; n++) {
          markerPositions[n]= markerPositions[n] / lag;
          if (nlhs >= 4) {
              char mrkType;
              char buffer [50];
              int length;
              double value;
              
              
              if(markerValues[n] >= 0.0) {
                  mrkType = 'S';
                  value = markerValues[n];
              } else {
                  mrkType = 'R';
                  value = -markerValues[n];
              }
          
              length=sprintf (buffer, "%c%3.0f", mrkType, value);
          
                
                mxSetCell(OUT_MRK_TOE, n, mxCreateString(buffer));
      /*	    pMrkToe[n]= ((*pszDesc =='R') ? -1 : 1) * atoi(pszDesc+1); */
          }
          if (nlhs == 5) { 
              if(markerValues[n] >= 0.0) {
                  mxSetCell(OUT_MRK_DESC, n, mxCreateString("Stimmulus"));
              } else {
                  mxSetCell(OUT_MRK_DESC, n, mxCreateString("Response"));
              }
            
          }
        }
        
        free(markerPositions);
        free(markerValues);

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
      connected = 0;
      
      emoComp_hostname = (char *) malloc(MAX_CHARS);
      /* getting the hostname for the new connection */
      pArray = mxGetField(IN_STATE, 0, "hostname");
      mxGetString(pArray, emoComp_hostname, MAX_CHARS);
      
      /* try reconnecting till we get a new connection */
      while(!aemo_connect(emoComp_hostname)){
        printf("connecting failed, trying again\n");
      }
      
      /* cleaning things up */
      free(emoComp_hostname);
      connected = 1;
    } else {
      printf("getData didn't work, closing connection, returning -2\n ");
      /* close the connection and clean everything up */
      aemo_close();
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
  free(pSrcDouble);

}

/************************************************************
 *
 * Initialize Connection
 *
 ************************************************************/

static void aemo_close()
{
  /* close the trhead */  
  while(1 == isThreadRunning) {
    keepThreadAlive = 0;
  }
  CloseHandle(threadHandle);
  threadHandle = NULL;
  
  if(NULL != hLib) {
    deleteLibrary(hLib);
    hLib = NULL;
  }
  

  
  EE_EngineDisconnect();
	EE_EmoEngineEventFree(eEvent);
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
static void aemo_assert(bool condition,const char *text) {
  if(0 == condition) {
    aemo_close();
    
    mexErrMsgTxt(text);
  }
}

static DWORD WINAPI readParallelport(LPVOID lpParameter) {
  int lastMarkerValue;  
  int value;
  
  
  
  lastMarkerValue = 0;
  
  while(1 == keepThreadAlive) {
    value = (int)Inp32(parallelportAddr);
    if(value != lastMarkerValue && 0 != value) {
      EE_DataSetMarker(0, value); 
      //printf("Thread is alive and got Marker %i", value);
    }
    lastMarkerValue = value;
    Sleep(0);
  }
  
  isThreadRunning = 0;
  return 0;
}
