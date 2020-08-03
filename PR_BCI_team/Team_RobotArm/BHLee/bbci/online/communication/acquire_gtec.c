/*
 * acquire_gtec.c
 *
 * This mex file gets the data from the gtec controllers. For more information see
 * acquire_gtec.m.
 * 
 * The cases for the function calls:
 * 1. Initialisation:
 *     state = acquire_gtec(sampling_freq, server); 
 *     state = acquire_gtec(sampling_freq, server, sampleFilter);
 *     state = acquire_gtec(sampling_freq, server, bFilter, aFilter);
 *     state = acquire_gtec(sampling_freq, server, sampleFilter, bFilter, aFilter);
 *   
 *    This function call sets up the values for the data acquisition.
 *    Arguments:
 *      sampling_freq :        The frequency to use for the sampling of the data in hz
 *      server:               The server for the data (This value is not used. It occurs
 *                                                      because we need to be compatible
 *                                                      with acquire_bv.m)
 *      sampleFilter:         The down sample filter if the recorder provides
 *                            the samples in a higher frequency
 *      aFilter, bFilter:     The arguments for an FIR filter
 *    Return:
 *        The state object where the user can change some settings
 * 2. data aquisition:
 *    [data, blockno, marker_pos, marker_token, marker_description]
 *      = acquire_gtec(state);
 *  
 *    This call gets a block of data from the controller.
 *    Arguments:
 *      state:                The state for the data aquistion.
 *    
 *    Return:
 *     Every return argument except data is optional.
 * 3. Stoping:
 *    acquire_gtec('close');
 *
 *    Closes the controller and deletes the variables
 *
 * 2008/12/19 - Max Sagebaum
 *               - file created 
 * 2009/09/18 - Max Sagebaum 
 *               - fixed a possible bug: If the channel selection was changed 
 *                  you got an invalid memory access exception            
 * 
 * (c) Fraunhofer FIRST.IDA 2008
 */

#include <windows.h>
#include <math.h>

/* The filter and buffer */
#include "filter.h"
#include "buffer.h"

/* The includes for the gTec controllers */
#include "gUSBamp.h"
#pragma comment(lib,"gUSBamp.lib")

#include "mex.h"

/* The data we need for each gTec controller */
typedef struct _ControllerData
{
  HANDLE hdev;          /* The handle to the controller */
  HANDLE dataEvent;     /* The event for the data access */
  OVERLAPPED ov;        /* The structure for the data access */
  BYTE *buffer;         /* The buffer for the data from the controller */
} ControllerData;

/* The arguments for the acquire thread */
typedef struct _AGTThreadArgs
{
  int deviceCount;              /* The number of controllers we have */
  ControllerData *controllers;  /* The data for each controller */
  BOOL* acquireThreadRunning;   /* The flag if the acquire thread is running or if the thread should stop */
} AGTThreadArgs;

static int BUFFER_SIZE = 9*60;          /* The size of the internal buffer in seconds */ 
static double BASE_FREQUENCY = 1200;    /* The base frequency for the controllers */
static int CONTROLLER_CHANNELS = 17;    /* The number of channels each controller has */
static const int MAX_CHARS = 1024;      /* maximum size of hostname */

/* The names for the structure we create */
static const char *FIELD_NAMES[] = {
  "clab", "lag", "orig_fs", "scale", "block_no","chan_sel","reconnect","hostname","fir_filter"
};
static const int NUMBER_OF_FIELDS = (sizeof(FIELD_NAMES)/sizeof(*FIELD_NAMES));

/*
 * This variable discribes the time in ms of the buffer inside the gtec 
 * controller.
 */
static int BLOCK_SIZE = 40; 

static int deviceCount = 0;           /* the number of gtec controllers */
static ControllerData *controllers;   /* Data for each gtec controller  */

/* The variables for the buffer in the gtec controller (called gTecBuffer) */
static int scanlines;                 /* the number of scans the gTecBuffer can hold */
static int bufferSize;                /* the size of the gTecBuffer in bytes */
static int scanlineSize;              /* the size of one scanline in bytes */

static HANDLE acquireThread;          /* The handle for the thread which acquires the data */
static BOOL acquireThreadRunning;     /* A flag for the thread: It is used to show the thread that it
                                       * should stop and inside the loop of the thread
                                       * to show the normal programm that the thread has 
                                       * stopped */

/* placeholders for the functions */
static void agt_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
static int agt_initGTEC();
static void agt_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
ULONG WINAPI agt_acquireLoop(void* pArg);
static void agt_close();
static void agt_assert(bool aValue,const char *message);
static void printError();

/*********************************************
 *
 * This funciton calculates the size of the buffer for given timeperiod and
 * sampling rate.
 *  Input:  time:             The time in ms the buffer can hold
 *          samplingRate:     The sampling rate of the gtec controller
 *
 *  Output: The size of the buffer for one gtec controller in scanlines
 *
 *********************************************/
static int calcBufferLength(int time, int samplingRate) {
  double size;
  
  size = (double)time * (double)samplingRate * 0.001;
  size = ceil(size);
  /* The controller cannot hold more than 512 scans */
  if(size > 512) {
    size = 512;
  }
  
  return (int)size;
}

/*********************************************
 *
 * The entry point for matlab.
 * The method checks for the parameters and decides which subrutine has to
 * be called. It also checks if the function is in the right state to perform 
 * the action.
 *
 *********************************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Check arguments and set mode and parameters */
  if (nrhs == 1 && mxIsStruct(prhs[0])) { /* data acquisition */
    agt_assert(nlhs < 6, "five output arguments maximum");
    agt_assert(nrhs == 1, "exactly one input argument required");
    agt_assert(mxIsStruct(prhs[0]), "input argument must be struct");
    if (0 == deviceCount) {
      mexWarnMsgTxt("acquire_gtec: open a connection first!");
    } else {
      agt_getdata(nlhs, plhs, nrhs, prhs);
    }
  } else if ((nrhs == 0) || (nrhs == 1 && mxIsChar(prhs[0]))) { /* close */
    agt_assert(nlhs == 0, "no output argument expected");
    if (0 == deviceCount) {
        mexWarnMsgTxt("acquire_gtec: open a connection first!");
    } else {
        agt_close(nlhs, plhs, nrhs, prhs);
    }
  } else {/* init */
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
      if (0 != deviceCount) {
        mexWarnMsgTxt("acquire_gtec: connection is still open!");
        mexWarnMsgTxt("acquire_gtec: closing the connection!");

        agt_close(nlhs, plhs, nrhs, prhs);
      }

      agt_init(nlhs, plhs, nrhs, prhs);
    } else { 
      mexWarnMsgTxt("acquire_gt: called with illegal parameters\n");
    }
  }
}

/**********************************************
 *  
 * The init function creates everything which is needed to get the datat from
 * the gTec controllers. It creates the structure with some parameters and 
 * information the user can acces and change in matlab. It creates the filters
 * for the IIR and FIR filtering and the buffer for the storage of the acquired
 * data from the gTec controller.
 * INPUT: nrhs      - The number of arguments in the matlab call
 *        prhs      - The structures in the matlab call
 *                    prhs[0] == frequency
 *                    prhs[1] == hostname
 *                    prhs[2,3,4] arguments for the filters (optional)
 * OUTPUT: nlhs     - The number of return variables in the matlab call
 *         plhs     - The pointers to the structures in the matlab call
 *                    plhs[0] == state
 **********************************************/
static void agt_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int frequency;
  int lag;
  mxArray* outStruct;
  const mxArray *RESAMPLE_FILTER;

  char *hostname = 0, *default_hostname= "localhost";
  
  /* read the frequency */
  {
    const mxArray *IN_FS;
    
    IN_FS = prhs[0];
    agt_assert(mxIsNumeric(IN_FS), "First argument must be real scalar.");
    agt_assert(mxGetM(IN_FS) * mxGetN(IN_FS) == 1, 
       "First argument must be real scalar.");
    frequency = (int)mxGetScalar(IN_FS);
    lag = (int) (BASE_FREQUENCY / frequency);

    /* check if BASE_FREQUENCZ / frequency is a natural number */
    agt_assert(lag * frequency == BASE_FREQUENCY," The base frequency has to be a multiple of the requested frequency.");
  }
  
  /* We do not need the second argument. We currently can only access a 
   * gtec controller from the local pc */
  {
    const mxArray *IN_HOSTNAME;
    
    if (nrhs >= 2) {
      IN_HOSTNAME = prhs[1];
      agt_assert(mxIsChar(IN_HOSTNAME), "Second argument must be a string.");
      hostname = (char *) malloc(MAX_CHARS);
      mxGetString(IN_HOSTNAME, hostname, MAX_CHARS);
    } else {
      hostname = (char*) malloc(strlen(default_hostname) + 1);
      strcpy(hostname, default_hostname);
    }  
  }
  
  /* init the gtec controller*/
  if(0 == agt_initGTEC()) {
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    return;
  }
  
  /* init the ring buffer for the raw data */
  {
    int bufferLines = BUFFER_SIZE * (int)BASE_FREQUENCY; 
    agt_assert(bufferCreate(bufferLines, deviceCount, CONTROLLER_CHANNELS),
            "The buffer for the data could not be created.");
  }

  /* create the filters */
  {
    bool iirFilterSet;
    bool resampleFilterSet;
    int i;

    const mxArray *IN_B_FILTER;
    const mxArray *IN_A_FILTER;    
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
        agt_close(nlhs, plhs, nrhs, prhs);
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      if(mxGetN(RESAMPLE_FILTER) != lag) {
        mexErrMsgTxt(" resample filter has to correspondent with the sampling rate.");
        agt_close(nlhs, plhs, nrhs, prhs);
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
    
      /* we have to copy the matlab structure into a new one other wise the use of a own fir 
       * filter will produce a matlab error */
      filter = mxGetPr(RESAMPLE_FILTER);
      RESAMPLE_FILTER = mxCreateDoubleMatrix(1, lag, mxREAL); 
      memcpy(mxGetPr(RESAMPLE_FILTER),filter,lag * sizeof(double));
      filter = mxGetPr(RESAMPLE_FILTER);

      filterFIRCreate(filter, lag,CONTROLLER_CHANNELS * deviceCount);
    } else {
      double* filter;

      /* the defalut filter will only take the last value from each block  */
      RESAMPLE_FILTER = mxCreateDoubleMatrix(1, lag, mxREAL); 
      filter = mxGetPr(RESAMPLE_FILTER);


      for(i = 0; i < lag;++i) {
        filter[i] = 0.0;
      }
      filter[lag - 1] = 1.0;

      filterFIRCreate(filter,lag, CONTROLLER_CHANNELS * deviceCount);
    }
    

    /* create the filter buffers */
    if(iirFilterSet) {
      /* check the dimensions  */
      if(mxGetM(IN_B_FILTER) * mxGetM(IN_A_FILTER) != 1) {
        mexErrMsgTxt(" bFilter and aFilter has to be a vector.");
        agt_close(nlhs, plhs, nrhs, prhs);
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }
      if(mxGetN(IN_B_FILTER) != mxGetN(IN_A_FILTER)) {
        mexErrMsgTxt(" bFilter and aFilter must have the same size.");
        agt_close(nlhs, plhs, nrhs, prhs);
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
      }

      filterIIRCreate(mxGetPr(IN_A_FILTER), mxGetPr(IN_B_FILTER), mxGetN(IN_B_FILTER), CONTROLLER_CHANNELS * deviceCount);
    } else {
      filterIIRCreate(NULL,NULL,1,CONTROLLER_CHANNELS * deviceCount); /* create the default IIR filter */   
    }
  }
  
  /* create the output structure */
  outStruct = mxCreateStructMatrix(1, 1, NUMBER_OF_FIELDS, FIELD_NAMES);
  plhs[0] = outStruct;

  /* some simple labels */
  mxSetField(outStruct, 0, "orig_fs", mxCreateDoubleScalar(BASE_FREQUENCY));
  mxSetField(outStruct, 0, "lag", mxCreateDoubleScalar(lag));
  mxSetField(outStruct, 0, "block_no", mxCreateDoubleScalar(-1.0));
  mxSetField(outStruct, 0, "reconnect", mxCreateDoubleScalar(0));
  mxSetField(outStruct, 0, "hostname", mxCreateString(hostname));
  mxSetField(outStruct, 0, "fir_filter", (mxArray *)RESAMPLE_FILTER);

  /* create the channel labels */
  {
    mxArray *channelNames;
    int n;

    char buffer[20];
    int myInteger = 10;


    channelNames = mxCreateCellMatrix(1, CONTROLLER_CHANNELS * deviceCount);    
    for (n = 0; n < CONTROLLER_CHANNELS * deviceCount; n++) {
      sprintf(buffer, "%i", n + 1);
      mxSetCell(channelNames, n, mxCreateString(buffer));
    }
    mxSetField(outStruct, 0, "clab", channelNames);
  }

  /* create the channel scale factors */
  {
    mxArray *scaleFactors;
    double *scaleFactorsP;
    int n;

    scaleFactors = mxCreateDoubleMatrix(1, CONTROLLER_CHANNELS * deviceCount, mxREAL);
    scaleFactorsP = mxGetPr(scaleFactors);      
    for (n = 0; n < CONTROLLER_CHANNELS * deviceCount; n++) {
      scaleFactorsP[n] = 1.0;
    }
    mxSetField(outStruct, 0, "scale", scaleFactors);
  }

  /* create the channel indices */
  {
    mxArray *channelIndices;
    double *channelIndicesP;
    int n;

    channelIndices = mxCreateDoubleMatrix(1, CONTROLLER_CHANNELS * deviceCount, mxREAL);
    channelIndicesP = mxGetPr(channelIndices);
    for (n = 0;n<CONTROLLER_CHANNELS * deviceCount;n++) {
        channelIndicesP[n] = n+1;
    }
    mxSetField(outStruct, 0, "chan_sel", channelIndices);
  }
  
  /* start the thread for the data acquisition */
  {
    AGTThreadArgs *params;
    
    params = (AGTThreadArgs*)malloc(sizeof(AGTThreadArgs));
    params->deviceCount = deviceCount;
    params->controllers = controllers;
    params->acquireThreadRunning = &acquireThreadRunning;
	  acquireThreadRunning = TRUE;
    acquireThread = CreateThread( NULL, 0, &agt_acquireLoop, (LPVOID)params, CREATE_SUSPENDED, NULL);
    SetThreadPriority(acquireThread, THREAD_PRIORITY_TIME_CRITICAL );
    ResumeThread(acquireThread); 
  }
}

/**********************************************
 *  
 * The function looks for the gtec controllers which are connected to this 
 * pc and sets all values in controllers
 * INPUT: -
 * RETURN: 0 if the controllers could not be initialized
 *         1 if everything is ok
 *
 **********************************************/
static int agt_initGTEC(){
  /* Calculate the value for the buffersize with the default frequency */
  
  HANDLE hdev;
  int i, contrPos;
  DAC myDAC;
  
  /* set the basic values for every controller */
  scanlines = calcBufferLength(BLOCK_SIZE,(int)BASE_FREQUENCY);
  scanlineSize = CONTROLLER_CHANNELS * sizeof(float);  /* this is only the size for one controller */
  bufferSize = scanlines * scanlineSize;
  
  /* find the divices connected to the pc */
  for(i = 1; i < 16; ++i) {
    hdev = GT_OpenDevice(i);
    
    if(NULL != hdev) {
      // found a divice
      ++deviceCount;
      mexPrintf("Found a device at the position %i.\n",i);
      GT_CloseDevice(&hdev);
    } 
  }
  
  /* create the array for the controllers and initialize them*/
  controllers = malloc(sizeof(ControllerData) * deviceCount);
  contrPos = 0;
  for(i = 1; i < 16; ++i) {
    hdev = GT_OpenDevice(i);
    
    if(NULL != hdev) {
      /* found a divice */
      controllers[contrPos].hdev = hdev;
      controllers[contrPos].dataEvent = CreateEvent(NULL,FALSE,FALSE,NULL);
      controllers[contrPos].ov.hEvent = controllers[contrPos].dataEvent;
      controllers[contrPos].ov.Offset = 0;
      controllers[contrPos].ov.OffsetHigh = 0;
      controllers[contrPos].buffer = (BYTE*)malloc(HEADER_SIZE + bufferSize);
      
      /* init the device */
      if(FALSE == GT_SetBufferSize(hdev,scanlines)) {
        printError();
        return 0;
      }
      if(FALSE == GT_SetSlave(hdev, 0 != contrPos)) { /* we assume that the first controller is the master */
        printError();
        return 0;
      }
      if(FALSE == GT_SetSampleRate(hdev, (int)BASE_FREQUENCY)) {
        printError();
        return 0;
      }
      
      /*if(FALSE == GT_EnableTriggerLine(hdev, true)) {
        printError();
        return 0;
      }*/
      
      if(FALSE == GT_EnableSC(hdev,TRUE)) {
        printError();
        return 0;
      }

      if(FALSE == GT_SetMode(hdev,M_IMPEDANCE)) {
         printError();
         return 0;
      }
      
      /* uncomment for debugging
       * produces sine waves
       */ 
      
      /*myDAC.Offset = 2047;
      myDAC.WaveShape = WS_SINE;
      myDAC.Frequency = 2;
      myDAC.Amplitude = 100;
      GT_SetDAC(hdev,myDAC);
      if(FALSE == GT_SetMode(hdev,M_CALIBRATE)) {
        printError();
        return 0;
      }
       */
       
      ++contrPos;
    } 
  }
  
  /* Start the devices in reverse order(slaves first) */
  for(i = deviceCount - 1; i >= 0;--i) {
    if(FALSE == GT_Start(controllers[i].hdev)) {
      printError();
      return 0;
    }
  }
  
  return 1;
}

/**********************************************
 *  
 * Reads the data from the buffer. Filters the data and 
 * returns it to matlab.
 * INPUT: nrhs      - The number of arguments in the matlab call
 *        prhs      - The structures in the matlab call
 *                    prhs[0] == state
 * OUTPUT: nlhs     - The number of return variables in the matlab call
 *         plhs     - The pointers to the structures in the matlab call
 *                    plhs[0 - 4] == the data structures
 *
 **********************************************/
static void agt_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {  
  if(TRUE == acquireThreadRunning){/* Filter the data and create the output values */
    mxArray* pArray;
    int nChans_orig, lag, nPoints, nChans_sel;
    double *chan_sel, *scale, *pDst0;
    
    
    double* dataStart;
    int numberOfScanlines;
    /* get the size of the data and the position */
    bufferRead(&dataStart,&numberOfScanlines);
    
    /* get the values form the state */
    pArray = mxGetField(prhs[0], 0, "clab");
    nChans_orig = mxGetN(pArray);

    pArray = mxGetField(prhs[0], 0, "lag");
    lag = (int) mxGetScalar(pArray);

    pArray = mxGetField(prhs[0], 0, "chan_sel");
    chan_sel = mxGetPr(pArray);
    nChans_sel = mxGetN(pArray);

    /* check for the new resample filter */
    pArray = mxGetField(prhs[0], 0, "fir_filter");
    if(mxGetM(pArray) != 1) {
      mexErrMsgTxt("new resample filter has to be a vector.");
      agt_close(nlhs, plhs, nrhs, prhs);
      plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
      return;
    }
    if(mxGetN(pArray) != lag) {
      mexErrMsgTxt("new resample filter has to correspondent with the sampling rate.");
      agt_close(nlhs, plhs, nrhs, prhs);
      plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
      return;
    }
    filterFIRSet(mxGetPr(pArray));

    /* the output block number */
    if(nlhs >= 2) {
      plhs[1] =
        mxCreateDoubleScalar(1); //TODO add a counter 
    }

    /* construct the data output matrix. */
    nPoints = (getFIRPos() + numberOfScanlines)/lag;
    plhs[0] = mxCreateDoubleMatrix(nPoints, nChans_sel, mxREAL);

    pArray = mxGetField(prhs[0], 0, "scale");
    scale = mxGetPr(pArray);
    pDst0= mxGetPr(plhs[0]);

    /* filter the data with the filters */
    
    filterData(dataStart,numberOfScanlines ,pDst0,nPoints, chan_sel, nChans_sel, scale);
    
    /* if markers are also requested, construct the appropriate output
       matrices */ 
    /* no markers -> return empty matrix */
    if (nlhs >= 3) {      
      plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
      
      if (nlhs >= 4) { 
        plhs[3] = mxCreateDoubleMatrix(0, 0, mxREAL);
        
        if (nlhs == 5) {
          plhs[4] = mxCreateCellMatrix(0, 0);             
        }
      }      
    }
  } else {
    mexPrintf("Timeout occured. Check the sync connector.\n Closing connection.");        
    mexPrintf("acquire thread closed. Closing connection.\n");
    agt_close(0,NULL,0,NULL);
  }
}

/**********************************************
 *  
 * The function for the acquire thread. It gets the data from the gTec
 * controllers and puts it to the buffer.
 * INPUT: pArg      - A pointer to a AGTThreadArgs structure.
 *                    which holds the arguments for this thread.
 *
 **********************************************/
ULONG WINAPI agt_acquireLoop(void* pArg) {
  ControllerData *controllers;      /* The data for each controller */
  int deviceCount;                  /* The number of controllers */
  BOOL *acquireThreadRunning;       /* A pointer to the variable which holds if the thread should keep on running */
  
  /* The values for the loop */
  int dwBytesReceived;              
  int curContrPos; 
  DWORD dwOVret;                    

  /* Set the values form the args structure */
  AGTThreadArgs *params = (AGTThreadArgs*)pArg;
  controllers = params->controllers;
  deviceCount = params->deviceCount;
  acquireThreadRunning = params->acquireThreadRunning;
  
  free(params); /* We do not need the args structure form here on */
  params = NULL;

  while(TRUE == *acquireThreadRunning) {
    /* send the data request for each controller */
    for(curContrPos = deviceCount - 1; curContrPos >= 0; --curContrPos) {
      /* set the overlapped structure for the data access*/
      ResetEvent(controllers[curContrPos].dataEvent);
      controllers[curContrPos].ov.hEvent = controllers[curContrPos].dataEvent;
      controllers[curContrPos].ov.Offset = 0;
      controllers[curContrPos].ov.OffsetHigh = 0;

      /* Send the data request*/
      if(FALSE == GT_GetData(controllers[curContrPos].hdev, controllers[curContrPos].buffer,HEADER_SIZE + bufferSize,&controllers[curContrPos].ov)) {
        *acquireThreadRunning = FALSE;
        printError();
      }
    }

    /* wait for the data of each controller */
    for(curContrPos = deviceCount - 1; curContrPos >= 0; --curContrPos) {
      int numberOfScanlines;

      /* wait for the data*/
      dwOVret = WaitForSingleObject(controllers[curContrPos].dataEvent,1000);
      if(dwOVret == WAIT_TIMEOUT)
      {
        /* we have a timeout. This occours in most case when the sync cable is not properly
         * connected */
        GT_ResetTransfer(controllers[curContrPos].hdev);
        *acquireThreadRunning = FALSE;
        continue;
      }

      /* get the number of bytes written to the buffer */
      dwBytesReceived = 0;
      GetOverlappedResult(controllers[curContrPos].hdev,&controllers[curContrPos].ov,&dwBytesReceived,FALSE);
      dwBytesReceived -= HEADER_SIZE; 
      numberOfScanlines = dwBytesReceived/(scanlineSize);
      
      /* write the data from the controller to the ring buffer */
      bufferWrite((float*)(controllers[curContrPos].buffer + HEADER_SIZE),numberOfScanlines, curContrPos);
    }
  }
  /* used for debugging */
  /* Beep(440,100); */
 
  return 0;
}

/**********************************************
 *  
 * Frees all resources which are needed by the methods and
 * stops the acquire thread.
 *
 **********************************************/
static void agt_close() {
  int curContrPos;

  /* stop the acquireThread and wait until it has stopped*/
  acquireThreadRunning = FALSE;
  WaitForSingleObject(acquireThread,INFINITE);
  
  /* delete the filters and buffers */
  filterClose();
  bufferClose();
  
  /* delete the structures */
  for(curContrPos = deviceCount - 1; curContrPos >= 0; --curContrPos) {
    ControllerData curContr = controllers[curContrPos];
    
    GT_Stop(curContr.hdev);
    CloseHandle(curContr.dataEvent);
    free(curContr.buffer);
  }
  
  free(controllers);
  deviceCount = 0;
}

/************************************************************
 *
 * checks for errors and does some cleanup before returning to matlab
 * INPUT: condition       - The condition for the assert. Everything is ok
 *                          if condition is equal to one.
 *        text            - The text which is printed when condition ist not equal to one
 *
 ************************************************************/
static void agt_assert(bool condition,const char *text) {
  if(0 == condition) {
    agt_close();
    
    mexErrMsgTxt(text);
  }
}

/*************************************************************
 *
 * Prints the error text after a failed command for the gtec controller.
 *
 *************************************************************/
static void printError() {
  WORD errorCode;
  char *errorText = NULL;
  GT_GetLastError(&errorCode,errorText);
  
  mexPrintf("The error was: Error(%i): %s.\n",errorCode,errorText);
}

