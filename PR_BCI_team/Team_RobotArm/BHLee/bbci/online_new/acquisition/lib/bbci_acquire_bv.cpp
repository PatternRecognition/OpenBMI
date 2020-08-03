/*
  bbci_acquire_bv.c

  This file defines a mex-Function to communicate with the brainvision
  server via matlab. The execution pathes are:
  
  1. state = bbci_acquire_bv('init', state); 
  2. state = bbci_acquire_bv('init', param1, value1, param2, value2, ...); 
  3. [data] = acquire_bv(state);
  4. [data, marker_time] = bbci_acquire_bv(state);
  5. [data, marker_time, marker_descr] = bbci_acquire_bv(state);
  6. [data, marker_time, marker_descr, state] = bbci_acquire_bv(state);
  7. bbci_acquire_bv('close'); 
  8. bbci_acquire_bv('close', DUMMY); 
 
  The first and the second call creates a connection to the brainvision server.
  The third to sixth call recevie data from the server. The last call 
  closes the connection to the server.
  
  NOTE: We observed a data loss when bbci_acquire_bv is called
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
 
 
 2011/04/19 - Max Sagebaum
              - copy from acquire_bv.c
              - changed handling for the new calling conventions
 2011/11/17 - Max Sagebaum
              - Added property list as an option for the conection to the 
                server
 2012/06/05 - Benjamin Blankertz
              - Corrected output of marker_time.pos to comply with the standard
                of the BBCI online toolbox.
              - Changed behavior for 'close' command without open connection
                from error to warning.
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
#include "../../../toolbox/fileio/msvc_stdint.h"
#else
#include <stdint.h>
#endif


#include "mex.h"
extern "C" {
  #include "brainserver.h"
}

#include "../../../online/winunix/winthreads.h"

/* All the handling for the filtering of the data */ 
#include "filter.h"

/*
 * DEFINES
 */

#define MAX_CHARS 1024 /* maximum size of hostname */

/*
 * GLOBAL DATA
 */
static int connected = 0;  /* 0 if we have no connection to the server
                            * 1 if we have a connection to the server */

static const char* FIELD_FS = "fs";
static const char* FIELD_HOST = "host";
static const char* FIELD_FILT_A = "filt_a";
static const char* FIELD_FILT_B = "filt_b";
static const char* FIELD_FILT_SUBSAMPLE = "filt_subsample";
static const char* FIELD_BLOCK_NO = "block_no";
static const char* FIELD_CHAN_SEL = "chan_sel";
static const char* FIELD_CLAB = "clab";
static const char* FIELD_LAG = "lag";
static const char* FIELD_SCALE = "scale";
static const char* FIELD_ORIG_FS = "orig_fs";
static const char* FIELD_RECONNECT = "reconnect";
static const char* FIELD_MARKER_FORMAT = "marker_format";

/*
 * FORWARD DECLARATIONS
 */ 

static void 
abv_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool isStructInit);

static void 
abv_getdata(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static double abv_filterDataIIR(double value, int channel) ;

static void 
abv_close();

static void abv_assert(bool condition,const char *text);

/* Some helper functions for the struct handling. */

static int getFieldNumber(mxArray *pStruct, const char *fieldname);

static int checkScalar(mxArray *pStruct, const char *fieldname, double defaultValue);
static double getScalar(const mxArray *pStruct, const char *fieldname);
static void setScalar(mxArray *pStruct, const char *fieldname, double value);

static int checkArray(mxArray *pStruct, const char *fieldname,int m, int n,
                       double* defaultValue);
static int checkArray(const mxArray *pStruct, const char *fieldname, int m, int n);

static double* getArray(const mxArray *pStruct, const char* fieldname);
static int getArrayM(const mxArray *pStruct, const char* fieldname);
static int getArrayN(const mxArray *pStruct, const char* fieldname);
static void setArray(mxArray *pStruct, const char *fieldname, double* array,
                     int m, int n);

static int checkString(mxArray *pStruct, const char *fieldname, const char* defaultValue);
static char* getString(const mxArray *pStruct, const char *fieldname);
static void setString(mxArray *pStruct, const char *fieldname, const char* value);

static void setStringCell(mxArray *pStruct, const char *fieldname, const char* array,
                          int m, int n);

static int compareString(const mxArray *pStruct, const char* compareString);

/*
 * This function will get the number of the field if the field doesn't exist
 * it will add the field to the struct.
 */
static int getFieldNumber(mxArray *pStruct, const char *fieldname){
  int fieldNumber = mxGetFieldNumber(pStruct, fieldname);
  
  if(-1 == fieldNumber) {
    fieldNumber = mxAddField(pStruct, fieldname);
    
    abv_assert(-1 != fieldNumber, "bbci_acquire_bv: Could not create field for struct.");
  }
  
  return fieldNumber;
}

/* 
 * This function checks if the given field is a scalar. If the field dosn't
 * exist it will set the default value.
 */
static int checkScalar(mxArray *pStruct, const char *fieldname, double defaultValue) {
  mxArray *pField;
  
  pField = mxGetField(pStruct, 0,fieldname);
  if(NULL == pField) {
    setScalar(pStruct,fieldname, defaultValue);
    return 1;
  } else {
    return 1 == mxIsDouble(pField)&& 1 == mxGetM(pField) && 1 == mxGetN(pField);
  }
}

/* 
 * Gets a scalar value from a structure field. Also checks if the field contains
 * only a scalar value.
 */
static double getScalar(const mxArray *pStruct, const char* fieldname) {
  mxArray *pField;
  
  pField = mxGetField(pStruct, 0,fieldname);
  
  abv_assert(NULL != pField, "bbci_acquire_bv: Field dosen't exist.");
  abv_assert(1 == mxIsDouble(pField), "bbci_acquire_bv: Field is no scalar"); 
  abv_assert(1 == mxGetM(pField), "bbci_acquire_bv: Field is no scalar"); 
  abv_assert(1 == mxGetN(pField), "bbci_acquire_bv: Field is no scalar"); 
  return mxGetScalar(pField);
}

/* 
 * Sets a scalar field in a struct.
 */
static void setScalar(mxArray *pStruct, const char *fieldname, double value) {
  int fieldNumber = getFieldNumber(pStruct, fieldname);
  mxSetFieldByNumber(pStruct, 0, fieldNumber, mxCreateDoubleScalar(value));
}

/*
 * Checks if the array has the correct dimension. If the field doesn't exist
 * it is set to the default value.
 */
static int checkArray(mxArray *pStruct, const char *fieldname,int m, int n,
                       double* defaultValue) {
  mxArray *pField;
  
  pField = mxGetField(pStruct, 0,fieldname);
  if(NULL == pField) {
    if(NULL != defaultValue) {
      setArray(pStruct,fieldname, defaultValue, abs(m), abs(n));
      return 1;
    } else {
      return 0;
    }
  } else {
    if(0 == mxIsDouble(pField)) {
      return 0;
    } else if(0 <= m && m != mxGetM(pField)) {
      return 0;
    } else if(0 <= n && n != mxGetN(pField)) {
      return 0;
    } else {
      return 1;
    }
  }
}

/*
 * Checks if the array has the correct dimension. If the field doesn't exist
 * it is set to the default value.
 */
static int checkArray(const mxArray *pStruct, const char *fieldname,int m, int n) {
  mxArray *pField;
  
  pField = mxGetField(pStruct, 0,fieldname);
  if(NULL == pField) {
      return 0;
  } else {
    if(0 == mxIsDouble(pField)) {
      return 0;
    } else if(0 <= m && m != mxGetM(pField)) {
      return 0;
    } else if(0 <= n && n != mxGetN(pField)) {
      return 0;
    } else {
      return 1;
    }
  }
}

/*
 * Reads an array from the struct.
 */
static double* getArray(const mxArray *pStruct, const char* fieldname) {
  mxArray *pField;
  pField = mxGetField(pStruct, 0,fieldname);
  
  abv_assert(NULL != pField, "bbci_acquire_bv: Field dosen't exist.");
  abv_assert(1 == mxIsDouble(pField), "bbci_acquire_bv: Field is no double type"); 
  return mxGetPr(pField);
}

/*
 * Gets the m dimension of an m x n Matrix.
 */
static int getArrayM(const mxArray *pStruct, const char* fieldname) {
  mxArray *pField;
  pField = mxGetField(pStruct, 0,fieldname);
  
  abv_assert(NULL != pField, "bbci_acquire_bv: Field dosen't exist.");
  return mxGetM(pField);
}

/*
 * Gets the n dimension of an m x n Matrix.
 */
static int getArrayN(const mxArray *pStruct, const char* fieldname) {
  mxArray *pField;
  pField = mxGetField(pStruct, 0,fieldname);
  
  abv_assert(NULL != pField, "bbci_acquire_bv: Field dosen't exist.");
  return mxGetN(pField);
}

/*
 * Sets the field in the struct to the given array.
 */
static void setArray(mxArray *pStruct, const char *fieldname, double* array,
                     int m, int n) {
  mxArray* pField;
  
  pField = mxCreateDoubleMatrix(m,n,mxREAL);
  memcpy(mxGetPr(pField), array, n*sizeof(double));

  int fieldNumber = getFieldNumber(pStruct, fieldname);
  mxSetFieldByNumber(pStruct, 0, fieldNumber, pField);
}

static int checkString(mxArray *pStruct, const char *fieldname, const char* defaultValue) {
  mxArray *pField;
  
  pField = mxGetField(pStruct, 0,fieldname);
  if(NULL == pField) {
    setString(pStruct, fieldname, defaultValue);
    return 1;;
  } else {
    return 1 == mxIsChar(pField);
  }
}

/* 
 * Gets a string value from a structure field. Also checks if the field contains
 * a string value.
 */
static char* getString(const mxArray *pStruct, const char *fieldname) {
  mxArray *pField;
  int stringSize;
  char* value;
  
  pField = mxGetField(pStruct, 0,fieldname);
  abv_assert(NULL != pField, "bbci_acquire_bv: Field dosen't exist.");
  abv_assert(1 == mxIsChar(pField), "bbci_acquire_bv: Field is not string.");
  
  stringSize = (mxGetM(pField) * mxGetN(pField) * sizeof(mxChar)) + 1;
  value = (char *) malloc(stringSize);
  mxGetString(pField,value, stringSize);
  
  return value;
}

/* 
 * Sets the field in a struct to the given string.
 */
static void setString(mxArray *pStruct, const char *fieldname, const char* value) {
  int fieldNumber = getFieldNumber(pStruct, fieldname);
  mxSetFieldByNumber(pStruct, 0, fieldNumber, mxCreateString(value));
}

/*
 * Sets the field of the struct to a cell array.
 */
static void setStringCell(mxArray *pStruct, const char *fieldname, const char* array,
                          int m, int n) {
  mxArray *pArray;
  int curIndex;
  char* curPos;
  
  curPos = (char*)array;
                            
  pArray = mxCreateCellMatrix(m, n);    
  for (curIndex = 0; curIndex < m*n; curIndex++) {
    mxSetCell(pArray, curIndex, mxCreateString(curPos));
    curPos += strlen(curPos) + 1;
  }
  
  int fieldNumber = getFieldNumber(pStruct, fieldname);
  mxSetFieldByNumber(pStruct, 0, fieldNumber, pArray);
}

static int compareString(const mxArray *pStruct, const char* compareString) {
  char* value;
  int stringSize;
  
  abv_assert(mxIsChar(pStruct),"bbci_acquire_bv: Can't compare a none string to a string");
  
  stringSize = (mxGetM(pStruct) * mxGetN(pStruct) * sizeof(mxChar)) + 1;
  value = (char *) malloc(stringSize);
  mxGetString(pStruct,value, stringSize);
  
  return strcmp (value, compareString);
}

/*
 * This functions adds a field to a struct. No check is performed because in 
 * mexFunction the check was done.
 */
static void addField(mxArray *pStruct, const mxArray *fieldName, const mxArray *fieldValue) {
  int stringSize = (mxGetM(fieldName) * mxGetN(fieldName) * sizeof(mxChar)) + 1;
  char *name = (char *) malloc(stringSize);
  mxGetString(fieldName,name, stringSize);  
  
  int fieldNumber = getFieldNumber(pStruct, name);
  
  mxArray *fieldValueCopy = mxDuplicateArray(fieldValue);
  mxSetFieldByNumber(pStruct, 0, fieldNumber, fieldValueCopy);
  
  free(name);
}

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
  /* first check for execution path 7 and 8 */
  if(mxIsChar(prhs[0]) && 0 == compareString(prhs[0], "close")) { 
    /* The user wants to close the connection */
      
    if(0 == nlhs) { /* no output arguments */
      /* Check if we have an open connection */
      if(0 == connected) {
        mexWarnMsgTxt("bbci_acquire_bv: no open connection to close!");
        return;
      } else {
        abv_close();
        return;
      }
    } else {
      mexErrMsgTxt("bbci_acquire_bv: no output arguments on close connection.");
    }
  }
  
  /* check for execution path 1 and 2 */
  if(1 <= nrhs && mxIsChar(prhs[0]) && 0 == compareString(prhs[0], "init")) {
    bool isStructInit = 2 == nrhs && mxIsStruct(prhs[1]); // check if we have path 1
    if(!isStructInit) { // if we have no struct init check if it is a property list
      bool isListInit = 0 == (nrhs - 1) % 2; // number of elements in the list must be a divisor of 2
      
      for(int i = 1; isStructInit && i < nrhs; i = i + 2) {
        // check if each second value in the list is a string
        isStructInit = isStructInit & mxIsChar(prhs[i]);
      }
      
      if(!isListInit) { // no valid init found
        mexErrMsgTxt("bbci_acquire_bv: init is only allowed with a struct or a property list as an argument.");
      }
      
    }
      
    if(1 == nlhs) { /* one output argument */
      if(0 != connected) {
        mexErrMsgTxt("bbci_acquire_bv: close the connection first!");
      } else {
        abv_init(nlhs, plhs, nrhs, prhs, isStructInit);
        return;
      }
    } else {
      mexErrMsgTxt("bbci_acquire_bv: only one output for a create connection.");
    }
  }
  
  /* check if we are in execution path 3 to 6 */
  abv_assert(4 >= nlhs, "bbci_acquire_bv: Four ouput arguments are maximum.");
  abv_assert(nrhs == 1, "bbci_acquire_bv: exactly one input argument required");
  abv_assert(mxIsStruct(prhs[0]), "bbci_acquire_bv: input argument must be struct");

  
  /* check if we are connect to the server */
  if(0 != connected) {
    abv_getdata(nlhs, plhs, nrhs, prhs);
  } else {
    mexErrMsgTxt("bbci_acquire_bv: open a connection first!");
  }
}

/************************************************************
 *
 * Initialize Connection
 *
 ************************************************************/

static void 
abv_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], bool isStructInit)
{
  int result;
  char *bv_hostname = 0;
  struct RDA_MessageStart *pMsgStart;
  
  mxArray* OUT_STATE;
  
  /* copy the params to the out state */
  if(isStructInit) {
    // init with struct
    OUT_STATE = mxDuplicateArray(prhs[1]);
  } else {
    // init with property list
    int dims[2] = {1,1};
    OUT_STATE = mxCreateStructArray(2, dims, 0, NULL);
    // arguments have beed checked in mexFunction
    for(int i = 1; i < nrhs; i = i + 2) {
      addField(OUT_STATE, prhs[i], prhs[i + 1]);
    }
  }
  
  /* check for the needed fields values 
   * if they don't exist we will set the default values */
  checkString(OUT_STATE,FIELD_HOST, "127.0.0.1");
  
  /* Get server name (or use default "brainamp") */
  bv_hostname = getString(OUT_STATE, FIELD_HOST);

  /* open connection */
  result = initConnection(bv_hostname,&pMsgStart);
  free(bv_hostname);  
    
  if (result == IC_OKAY) {
    /* construct connection state structure */
    int nChans, lag, n;
    double orig_fs;
    mxArray *pArray;
    char *pChannelName;
    double *chan_sel;
    double *filter_buffer_sub;
    double *filter_buffer_a;
    double *filter_buffer_b;
    int iirFilterSize;
    
    nChans = pMsgStart->nChannels;
    orig_fs = 1000000.0 / ((double) pMsgStart->dSamplingInterval);
    
    /* Check fs */
    checkScalar(OUT_STATE,FIELD_FS,orig_fs);
    
    lag = (int) (orig_fs / getScalar(OUT_STATE, FIELD_FS));
    
    abv_assert(lag * (int)getScalar(OUT_STATE,FIELD_FS) == (int)orig_fs,"bbci_acquire_bv: The base frequency has to be a multiple of the requested frequency.");
    
    /* Overwrite the following fields */
    setScalar(OUT_STATE,FIELD_ORIG_FS, orig_fs);
    setScalar(OUT_STATE,FIELD_LAG, lag);
    setScalar(OUT_STATE,FIELD_BLOCK_NO, -1.0);
     /* this odd hack is because pMsgStart contains several variably
       sized arrays, and this is the way to get the channel names 
     */
    setStringCell(OUT_STATE, FIELD_CLAB,(char *) ((double*) pMsgStart->dResolutions + nChans), 1, nChans);
    
    
    /* Check the following fields */
    checkString(OUT_STATE,FIELD_MARKER_FORMAT, "numeric");
    abv_assert(1 == checkScalar(OUT_STATE, FIELD_RECONNECT, 1), "bbci_acquire_bv: Reconnect is no scalar.");
    abv_assert(1 == checkArray(OUT_STATE, FIELD_SCALE, 1, nChans, pMsgStart->dResolutions), "bbci_acquire_bv: Scale is no array or has wrong size.");
    
    chan_sel = (double *) malloc(nChans*sizeof(double));
    for (n = 0;n<nChans;n++) {
        chan_sel[n] = n+1;
    }
    abv_assert(1 == checkArray(OUT_STATE, FIELD_CHAN_SEL, 1, -nChans, chan_sel), "bbci_acquire_bv: chan_sel is no array.");  
    free(chan_sel);
    
    /* Create the default filters */
    
    filter_buffer_sub = (double *) malloc(lag*sizeof(double));
    for(n = 0; n < lag; ++n) {
      filter_buffer_sub[n] = 1.0 / (double)lag;
    }
    filter_buffer_a = (double *) malloc(sizeof(double));
    filter_buffer_a[0] = 1.0;
    filter_buffer_b = (double *) malloc(sizeof(double));
    filter_buffer_b[0] = 1.0;
    
    /* check the filters */
    abv_assert(1 == checkArray(OUT_STATE, FIELD_FILT_SUBSAMPLE, 1, lag, filter_buffer_sub), "bbci_acquire_bv: Subsample filter is no array or has the wrong size.");
    abv_assert(1 == checkArray(OUT_STATE, FIELD_FILT_A, 1, -1, filter_buffer_a), "bbci_acquire_bv: IIR filter aSubsample filter has the wrong size.");
    abv_assert(1 == checkArray(OUT_STATE, FIELD_FILT_B, 1, -1, filter_buffer_b), "bbci_acquire_bv: Subsample filter has the wrong size.");
    
    /* free the default filters */
    free(filter_buffer_sub);
    free(filter_buffer_a);
    free(filter_buffer_b);
    
    /* check if the iir filters have the same size */
    iirFilterSize = getArrayN(OUT_STATE, FIELD_FILT_A);
    
    abv_assert(getArrayN(OUT_STATE, FIELD_FILT_B) == iirFilterSize, "bbci_acquire_bv: bFilter and aFilter must have the same size.");
    
    
    /* get the arrays for the filters and create the filters */
    filter_buffer_sub = getArray(OUT_STATE, FIELD_FILT_SUBSAMPLE);
    filter_buffer_a = getArray(OUT_STATE, FIELD_FILT_A);
    filter_buffer_b = getArray(OUT_STATE, FIELD_FILT_B);
    
    filterFIRCreate(filter_buffer_sub, lag,nChans);
    filterIIRCreate(filter_buffer_a, filter_buffer_b, iirFilterSize, nChans);
    
    connected = 1;
  }
  
  plhs[0] = OUT_STATE;
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
  
  /* init the input and output values */
  const mxArray* IN_STATE = prhs[0];
  mxArray* OUT_DATA = NULL;
  mxArray* OUT_MRK_TIME = NULL;
  mxArray* OUT_MRK_DESCR = NULL;
  mxArray* OUT_STATE = NULL;
    
  /* get the information from the state and obtain the data */
  lastBlock = (int)getScalar(IN_STATE, FIELD_BLOCK_NO);
  nChannels = getArrayN(IN_STATE, FIELD_CLAB);
  
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
    char* outputTypeDef;
    int outputType;
    double *pMrkToe;

    /* get necessary information from the current state */
    nChans_orig = getArrayN(IN_STATE, FIELD_CLAB);
    lag = (int) getScalar(IN_STATE,FIELD_LAG);

    nPoints = (getFIRPos() + pMsgData->nPoints)/lag;

    chan_sel = getArray(IN_STATE, FIELD_CHAN_SEL);
    nChans_sel = getArrayN(IN_STATE, FIELD_CHAN_SEL);
    
    /* check for the new resample filter */
    abv_assert(checkArray(IN_STATE, FIELD_FILT_SUBSAMPLE, 1, lag), "bbci_acquire_bv: Resample filter has to be a vector. Resample filter has to correspondent with the sampling rate.");
    filterFIRSet(getArray(IN_STATE, FIELD_FILT_SUBSAMPLE));
    
    /* construct the data output matrix. */
    OUT_DATA = mxCreateDoubleMatrix(nPoints, nChans_sel, mxREAL);
    
    pArray = mxGetField(IN_STATE, 0, "scale");
    scale = mxGetPr(pArray);
    pDst0= mxGetPr(OUT_DATA);
    pDstPosition = 0;

    /* convert the source data to double format */
    pSrcDouble = (double*)malloc(pMsgData->nPoints * nChans_orig * sizeof(double));
    if (ElementSize==2) {
        int16_t *pSrc = pMsgData->nData;
        for(n = 0; n != pMsgData->nPoints * nChans_orig; ++n) {
          pSrcDouble[n] = (double)pSrc[n];
        }
    } else if (ElementSize==4) {
        int32_t *pSrc = (int32_t*) pMsgData->nData;
        for(n = 0; n != pMsgData->nPoints * nChans_orig; ++n) {
          pSrcDouble[n] = (double)pSrc[n];
        }
    } else {
        mexErrMsgTxt("bbci_acquire_bv: Unknown element size");
    }
    
    /* filter the data with the filters */
    filterData(pSrcDouble,pMsgData->nPoints ,pDst0,nPoints, chan_sel, nChans_sel, scale);
    free(pSrcDouble);
    
    /* if markers are also requested, construct the appropriate output
       matrices */ 
    if (nlhs >= 2) {
      nMarkers = pMsgData->nMarkers;

      if (nMarkers > 0) {
        /* if markers existed, collect them */
        OUT_MRK_TIME = mxCreateDoubleMatrix(1, nMarkers, mxREAL);
        pMrkPos = mxGetPr(OUT_MRK_TIME);
        
        outputTypeDef = getString(IN_STATE, FIELD_MARKER_FORMAT);
        if(0 == strcmp ("numeric", outputTypeDef)) {
          outputType = 1;
        } else if(0 == strcmp ("string", outputTypeDef)) {
          outputType = 2;
        } else {
          mexErrMsgTxt("bbci_acquire_bv: Unknown ouput type.");
        }
        free(outputTypeDef);

        if (nlhs >= 3) {
          if(1 == outputType) { /* numeric */
            OUT_MRK_DESCR = mxCreateDoubleMatrix(1, nMarkers, mxREAL);
            pMrkToe = mxGetPr(OUT_MRK_DESCR);
          } else if(2 == outputType) {/* string */
            OUT_MRK_DESCR = mxCreateCellMatrix(1,nMarkers);
          } else {
            mexErrMsgTxt("bbci_acquire_bv: Unknown ouput type for output.");
          }
        }

        pMarker = (struct RDA_Marker*)((char*)pMsgData->nData + pMsgData->nPoints * nChans_orig * ElementSize);

        double origFs = getScalar(IN_STATE, FIELD_ORIG_FS);
        for (n = 0; n < nMarkers; n++) {
          pMrkPos[n]= ((double)pMarker->nPosition+1.0) * 1000.0 /origFs;
          pszType = pMarker->sTypeDesc;
          pszDesc = pszType + strlen(pszType) + 1;
          if (nlhs >= 3) {
            if(1 == outputType) { /* numeric */
              pMrkToe[n]= ((*pszDesc =='R') ? -1 : 1) * atoi(pszDesc+1);
            } else if(2 == outputType) {/* string */
              mxSetCell(OUT_MRK_DESCR, n, mxCreateString(pszDesc));
            } else {
              mexErrMsgTxt("bbci_acquire_bv: Unknown ouput type for output.");
            }
          }
          
          pMarker = (struct RDA_Marker*)((char*)pMarker + pMarker->nSize);
        }

      }
      else {
        /* no markers -> return empty matrix */
        OUT_MRK_TIME = mxCreateDoubleMatrix(0, 0, mxREAL);
        if (nlhs >= 3) {
          OUT_MRK_DESCR = mxCreateDoubleMatrix(0, 0, mxREAL);
        }
      }
    } /* end constructing marker outputs */
  }
  else {
    int nChans_sel;
    
    reconnect = (int) getScalar(IN_STATE, FIELD_RECONNECT);
    if(1 == reconnect) {
      printf("bbci_acquire_bv: getData didn't work, reconnecting ");

      /* only close the connection */
      closeConnection();
      connected = 0;
      
      bv_hostname = (char *) malloc(MAX_CHARS);
      /* getting the hostname for the new connection */
      pArray = mxGetField(IN_STATE, 0, "hostname");
      mxGetString(pArray, bv_hostname, MAX_CHARS);
      
      free(pMsgData);
      /* try reconnecting till we get a new connection */
      while(IC_OKAY != (result = initConnection(bv_hostname, &pMsgStart))){
        printf("bbci_acquire_bv: connecting failed, trying again\n");
        free(pMsgData);
      }
      
      /* cleaning things up */
      free(bv_hostname);
      free(pMsgStart);
      connected = 1;
    } else {
      printf("bbci_acquire_bv: getData didn't work, closing connection, returning -2\n ");
      /* close the connection and clean everything up */
      abv_close();
    }
    
    /* We have an error in the data transmition return an empty datablock. */
    pArray = mxGetField(IN_STATE, 0, "chan_sel");
    nChans_sel = mxGetN(pArray);

    OUT_DATA = mxCreateDoubleMatrix(0, nChans_sel, mxREAL);

    if (nlhs >= 2){OUT_MRK_TIME = mxCreateDoubleMatrix(0,0, mxREAL);};
    if (nlhs >= 3){OUT_MRK_DESCR = mxCreateDoubleMatrix(0,0, mxREAL);};
  }
  
  /* clone the state */
  if(nlhs >= 4) {OUT_STATE = mxDuplicateArray(IN_STATE);};
  
  plhs[0] = OUT_DATA;
  if(nlhs >=2) {
    plhs[1] = OUT_MRK_TIME;
  }
  if(nlhs >=3) {
    plhs[2] = OUT_MRK_DESCR;
  }
  if(nlhs >=4) {
    plhs[3] = OUT_STATE;
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
