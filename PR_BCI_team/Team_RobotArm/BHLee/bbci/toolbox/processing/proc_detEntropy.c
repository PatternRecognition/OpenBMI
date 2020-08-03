/*
  proc_detEntropy.c

  This file defines a mex-Function to calculate the entropy of a dataset.
  For the calling and data information see proc_getEntropy.m
  
 [H, C, I] = proc_detEntropy(x,scanLinePos,winSize, ..
                             'StepSize', stepSize, ..  \ only one of these 
                             'WinPos',winPos, ..       / two
                             'Concat,{0,1});   <-- optional 
 
  Arguments:
      x             - the data set
      scanLinePos   - an array with the position of the scanlines
      winSize       - a value with the size of the scan window
      stepSize      - a value with the offset for each window
      winPos        - an array with the starting positions of the window
                      positions
      Concat        - 0 or 1 (false or true)
   
  2008/04/22 - Max Sagebaum
                - file created 
  
  (c) Fraunhofer FIRST.IDA 2008
*/

#include <stdio.h>
#include <string.h>
#include "mex.h"

#include "proc_detEntropy_qtc.c"

/* the names for the arguments*/
const char *STEP_SIZE_ARG = "StepSize";
const char *WIN_POS_ARG = "WinPos";
const char *CONCAT_ARG = "Concat";

/* the values */
double *dataSet;
int dataSetSize;
double *scanLinePos;
int scanLinePosSize;
int windowSize;
int *windowPositions;
int windowPositionsSize;
bool concat;

/*
 * FORWARD DECLARATIONS
 */ 

void pde_init(int nrhs,const mxArray *prhs[]);

void pde_analyse( int nlhs, mxArray *plhs[]);

void pde_getBinaryString(double *dataSet, char *binaryString, int length,double scanLine);

void pde_call_qtc(char *data, int length, double *e, double *c, double *i);

void pde_cleanup();

void pde_assert(bool aValue,const char* text);


/************************************************************
 *
 * mexFunction
 *
 ************************************************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* First check of argument counts and output */
  pde_assert(nrhs >= 5 && nrhs <= 7, "five to seven input arguments required.");
  pde_assert(nlhs == 3, "Exactly three output arguments required.");
  /* do a cleanup to prevent a error if we killed the calculation during execution */
  releaseQtc();
    
  /* check the argument values and setup the */
  pde_init(nrhs,prhs);
  
  pde_analyse(nlhs,plhs);
  
  pde_cleanup();
}

/************************************************************
 *
 * Checks if the input values are valid and sets up the values
 * for the qtc calculation
 *
 ************************************************************/

void pde_init(int nrhs, const mxArray *prhs[])
{
  const mxArray *EEG_DATA;
  const mxArray *SCAN_LINE_POS;
  const mxArray *WIN_SIZE;
  const mxArray *STEP_SIZE;
  const mxArray *WIN_POS;
  const mxArray *CONCAT;
  
  bool isWinPosSet;
  bool initQtcSucced;
  int i,j;  /* temp counting value */
  int tempStepSize;
  double *tempArray;
  
  char *charBuf;            /* buffer for char reading (please free after usage) */
  mwSize charBufLength;     /* the size of the buffer */
  
  /* 
   * loading the time series
   */
  EEG_DATA = prhs[0];
  pde_assert(mxIsNumeric(EEG_DATA), "x must be a real scalar vector.");
  pde_assert(mxGetM(EEG_DATA) == 1, "x has to be a vector.");
  dataSetSize = mxGetN(EEG_DATA);
  dataSet = mxGetPr(EEG_DATA);
  
  /*
   * loading the scan line positions
   */  
  SCAN_LINE_POS = prhs[1];
  pde_assert(mxIsNumeric(SCAN_LINE_POS), "x must be a real scalar vector.");
  pde_assert(mxGetM(SCAN_LINE_POS) == 1, "x has to be a vector.");
  scanLinePosSize = mxGetN(SCAN_LINE_POS);
  scanLinePos = mxGetPr(SCAN_LINE_POS);
  
  
  /* 
   * loading the window size
   */
  WIN_SIZE = prhs[2];
  pde_assert(mxIsNumeric(WIN_SIZE), "winSize must be a real scalar.");
  pde_assert(mxGetM(WIN_SIZE) * mxGetN(WIN_SIZE) == 1, 
           "winSize must be a real scalar.");
  windowSize = (int)mxGetScalar(WIN_SIZE);
  pde_assert(windowSize <= dataSetSize,"winSize has to be less or equal to the size of the data set.");
  
  /*
   * loading the variable arguments
   */
  isWinPosSet = false;
  concat = false; /* set the default value */
  for(i = 3; (i + 1) < nrhs; i = i + 2) {
    /* get the name of the argument */
    pde_assert(mxIsChar(prhs[i]), "One of the argument specifiers is no string.");
    charBufLength = mxGetNumberOfElements(prhs[i]) + 1;
    charBuf = malloc(charBufLength * sizeof(char));
    if (mxGetString(prhs[i], charBuf, charBufLength) != 0) {
        free(charBuf); charBuf = 0;
        pde_cleanup();
        mexErrMsgTxt("Could not read argument.");
        return;
    }
    if(memcmp(charBuf,STEP_SIZE_ARG,charBufLength) == 0) {
      /*
       * loading the stepSize
       */
      pde_assert(isWinPosSet == false, "You can only set winPos or stepSize.");
      STEP_SIZE = prhs[i + 1];
      pde_assert(mxIsNumeric(STEP_SIZE), "stepSize must be a real scalar.");
      pde_assert(mxGetM(STEP_SIZE) * mxGetN(STEP_SIZE) == 1, 
           "stepSize musst be a real scalar.");
      tempStepSize = (int)mxGetScalar(STEP_SIZE);
	   pde_assert(tempStepSize >= 1, 
           "stepSize musst be greater or equal to 1.");
      
      /* the number of steps we can make */
      windowPositionsSize = 1 + (dataSetSize - windowSize) / tempStepSize; 
      windowPositions = malloc(windowPositionsSize * sizeof(int));
      
      for(j = 0; j < windowPositionsSize; ++j) {
        windowPositions[j] = j * tempStepSize;
      }
      
      isWinPosSet = true;
    } else if(memcmp(charBuf,WIN_POS_ARG,charBufLength) == 0) {
      /*
       * loading the window positions
       */
      pde_assert(isWinPosSet == false, "You can only set winPos or stepSize.");
      WIN_POS = prhs[i + 1];
      pde_assert(mxIsNumeric(WIN_POS), "winPos must be a real scalar vector.");
      pde_assert(mxGetM(WIN_POS) == 1, "pos must be a real scalar vector.");
      windowPositionsSize = mxGetN(WIN_POS);
      windowPositions = malloc(windowPositionsSize * sizeof(int));
      
      tempArray = mxGetPr(WIN_POS);
      for(j = 0; j < windowPositionsSize; ++j) {
        windowPositions[j] = (int)tempArray[j] - 1;/* matlab c index shift */
        if(windowPositions[j] < 0) {
            pde_assert(false,"first window(s) start too early. Check data length, win positions and win size."); 
        } else if(windowPositions[j] + windowSize > dataSetSize) {
            pde_assert(false,"last window(s) do not fit in data. Check data length, win positions and win size."); 
        }
        
      }
      
      isWinPosSet = true;
    } else if(memcmp(charBuf,CONCAT_ARG,charBufLength) == 0) {
      /*
       * loading the concat value
       */
      CONCAT = prhs[i + 1];
      pde_assert(mxIsNumeric(CONCAT), "concat must be a real scalar equal to 0 or 1.");
      pde_assert(mxGetM(CONCAT) * mxGetN(CONCAT) == 1, 
           "concat must be a real scalar equal to 0 or 1.");
      if(0 == mxGetScalar(CONCAT)) {
        concat = false;
      } else if( 1 == mxGetScalar(CONCAT)) {
        concat = true;
      } else {
        pde_assert(false, "concat must be a real scalar equal to 0 or 1.");
      }
    } else {
      pde_assert(false, "You can only set the arguments WinPos, StepSize and Concat.");
    }
    
    free(charBuf);
  }
  
  pde_assert(isWinPosSet, "You have to set the argument WinPos or StepSize.");
  
  /* init qtc*/
  if(concat) {
    initQtcSucced = initQtc(windowSize * scanLinePosSize);
  } else {
    initQtcSucced = initQtc(windowSize);
  }
  
    pde_assert(initQtcSucced,"Qtc could not create the data buffers");
  
} 

/************************************************************
 *
 * analyse the eeg data with qtc
 *
 ************************************************************/

void pde_analyse(int nlhs, mxArray *plhs[])
{  
  double *outDataH;         /* the return data of the matlab matrix as double array */
  double *outDataC;         /* the return data of the matlab matrix as double array */
  double *outDataI;         /* the return data of the matlab matrix as double array */
  
  char *analyseDataBlock;  /* one data block we will analyse */
  int  analyseDataBlockSize;
  
  int posWindowPositions;
  int posScanLinePos;
  int matrixIndex;
  
  if(concat) {
    /* we put every scanline into one string */
    analyseDataBlockSize = windowSize * scanLinePosSize;
  } else {
    /* we handle each scnaline separately */
    analyseDataBlockSize = windowSize;
  }
  analyseDataBlock = malloc(analyseDataBlockSize * sizeof(double));
  
  /* construct the data output matrix. */
  if(concat) {
    plhs[0] = mxCreateDoubleMatrix(1, windowPositionsSize, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, windowPositionsSize, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, windowPositionsSize, mxREAL);
  } else {
    plhs[0] = mxCreateDoubleMatrix(scanLinePosSize, windowPositionsSize, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(scanLinePosSize, windowPositionsSize, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(scanLinePosSize, windowPositionsSize, mxREAL);
  }
  outDataH = mxGetPr(plhs[0]); 
  outDataC = mxGetPr(plhs[1]); 
  outDataI = mxGetPr(plhs[2]); 

  for(posWindowPositions = 0; posWindowPositions < windowPositionsSize; ++posWindowPositions) {
    if(concat) {
      /* first get all the scanlines for the window */
      for(posScanLinePos = 0; posScanLinePos < scanLinePosSize; ++posScanLinePos) {
        pde_getBinaryString(dataSet + windowPositions[posWindowPositions], /* a pointer to the * positions of the dataSet set */
                            analyseDataBlock + (posScanLinePos * windowSize), /* a pointer to the * interval of data */
                            windowSize,
                            scanLinePos[posScanLinePos]);
      }
      
      /* calculate the entropy,... for the window */
       pde_call_qtc(analyseDataBlock,analyseDataBlockSize,
                        outDataH + posWindowPositions,
                        outDataC + posWindowPositions,
                        outDataI + posWindowPositions);
    } else {
      /* for each scanline we get a entropy value */
      for(posScanLinePos = 0; posScanLinePos < scanLinePosSize; ++posScanLinePos) {
        pde_getBinaryString(dataSet + windowPositions[posWindowPositions], /* a pointer to the * positions of the dataSet set */
                            analyseDataBlock,
                            windowSize,
                            scanLinePos[posScanLinePos]);
        
        matrixIndex = posScanLinePos + posWindowPositions * scanLinePosSize;
        pde_call_qtc(analyseDataBlock,windowSize,
                        outDataH + matrixIndex,
                        outDataC + matrixIndex,
                        outDataI + matrixIndex);
      }
    }
  }
  
  free(analyseDataBlock);analyseDataBlock = 0;
}

/************************************************************
 *
 * get the binary data string for the an eeg data window and
 * a scan line
 *
 ************************************************************/

void pde_getBinaryString(double *dataSet,char *binaryString, int length,double scanLine) {
  int i;
  
  for(i = 0; i < length; ++i) {
    if(scanLine <= dataSet[i]) {
      binaryString[i] = '1';
    } else {
      binaryString[i] = '0';
    }
  }  
}


/************************************************************
 *
 * call the qtc function (proc_detEntropy_qtc.c) and calculate the entropy, complexity, 
 * and information
 *
 ************************************************************/

void pde_call_qtc(char *data, int length, double *h, double *c, double *i) {
  double r;
  r = ftdSelAugment(data , length);
  c[0] = r;
  i[0] = invlogint(r);
  h[0] = i[0] / (double)length;
}


/************************************************************
 *
 * Free the space we have used and close the eeg file.
 *
 ************************************************************/

void pde_cleanup()
{
  /* do not free the other arrays - they are direct matlab pointers! */
  if(windowPositions != 0) {
    free(windowPositions); windowPositions = 0;
  }
  
  /* cleanup qtc */
  releaseQtc();
}

/************************************************************
 *
 * checks for errors and does some cleanup before returning to matlab
 *
 ************************************************************/
void pde_assert(bool aValue,const char *text) {
  if(!aValue) {
    pde_cleanup();
    
    mexErrMsgTxt(text);
  }
}
