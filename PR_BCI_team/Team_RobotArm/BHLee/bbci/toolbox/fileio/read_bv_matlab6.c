/*
  read_bv.c

  This file defines a mex-Function to load an eeg-File from a location.
  
  data = read_bv(file, HDR, OPT); 
  read_bv(file, HDR, OPT); / with opt.data and opt.dataPos set
 
  Arguments:
      file - Name of EEG file (.eeg) is appended)
      HDR  - Information about the file (read from the *.vhdr header file)
        .fs   - Sampling rate
        .nChans - Number of channels
        .nPoints - Number of data points in the file (optional)
        .scale  - Scaling factors for each channel
        .endian - Byte ordering: 'l' little or 'b' big
      OPT   - Struct with following fields
        .chanidx         - Indices of the channels that are to be read
        .fs              - Down sample to this sampling rate
        .filt_b          - Filter coefficients of IIR filter applied to raw data (b part) (optional)
        .filt_a          - Filter coefficients of IIR filter applied to raw data (a part) (optional)
        .filt_subsample  - Filter coefficients of FIR filter used for sub sampling (optional)
        .data            - A matrix where the data is stored (optional)
        .dataPos         - The position in the matrix[dataStart dataEnd fileStart fileend](optional)
 
 The filter parts of the OPT structure are optional fields.
 The default for the filt_subsample is a filter which takes the last value of
 filtered block e.g. [0 ... 0 1]
   
  2008/04/07 - Max Sagebaum
                - file created 
  2008/04/25 - Max Sagebaum
                - the char buffer was allocated with mxCalloc and freed 
                  this was wrong I changed it to malloc
  2008/05/20 - Max Sagebaum
                - checked for linux compatibly
                - fixed some errors in argument checking
  2008/06/20/ - Max Sagebaum
                - added option to load data into matrix
  
  (c) Fraunhofer FIRST.IDA 2008
*/

#include <stdio.h>
#include "mex.h"

/*
 * Define statements
 */

#define INT_16_SIZE 2 /* sizeof(short) */
#define mwSize int    /* not available in Matlab 6 */

/* the field names for HDR and OPT*/
const char *FS_FIELD = "fs";
const char *N_CHANS_FIELD = "nChans";
const char *N_POINTS_FIELD = "nPoints";
const char *SCALE_FIELD = "scale";
const char *ENDIAN_FIELD = "endian";
const char *CHAN_ID_X_FIELD = "chanidx";
const char *FILT_A_FIELD = "filt_a";
const char *FILT_B_FIELD = "filt_b";
const char *FILT_SUBSAMPLE_FIELD = "filt_subsample";
const char *DATA = "data";
const char *DATA_POS = "dataPos";

/* the handle for the eeg-file */
static FILE *eegFile;

/* hdr data values */
static double rawDataSamplingRate;
static int rawDataChannelCount;
static double *rawDataScale;
static char rawDataEndian;
static int rawDataPoints;

/* opt non optional values */
static double *optChannelSelect;
static int optChannelSelectCount;
static double optSamplingRate;

/* the values for the IIR filter  */
static int iirFilterSize = 0;            /* the size of the IIR filter  */
static int iirFilterOffset = 0;          /* we access the buffers toroidal  */
static double *bFilter;               /* the b part of the IIR filter  */
static double *aFilter;               /* the a part of the IIR filter  */

static double *xBuffer;               /* the buffer for the unfiltered values  */
static double *yBuffer;               /* the buffer for the filtered values  */

/* the values for the FIR filter  */
static double *firFilter;        /* a filter for the re sampling of the data  */
static int    firFilterPosition; /* the position in the filter  */
static int    firFilterSize;     /* the size of the filter  */
static double firFilterSum;      /* the sum of the values in the filter  */
static double *firFilterValues;  /* the current values for the re sampling  */

/* the positions of the samples when we write in a matrix*/
static double* dataPtr;
static int dataPtrSize;         /* the number of rows in the data */
static int dataStart;           /* the position of the first sample in the data*/
static int dataEnd;             /* the position of the last sample in the data*/
static int fileStart;           /* the first position of the data in the file*/ 
static int fileEnd;             /* the last position of the data in the file*/

/*
 * FORWARD DECLARATIONS
 */ 

static void rbv_init(int nrhs,const mxArray *prhs[]);

static bool 
rbv_readDataBlock(double *dataBlock, short* dataBuffer, int channelCount ,bool swap);

static void rbv_readData( int nlhs, mxArray *plhs[]);

static double rbv_filterDataIIR(double value, int channel) ;

static void rbv_cleanup();

static void rbv_assert(bool aValue,const char* text);


/************************************************************
 *
 * mexFunction
 *
 ************************************************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* First check of argument counts and output */
  rbv_assert(nrhs == 3, "Exactly three input arguments required.");
  rbv_assert(nlhs <= 1, "One or two output arguments required.");
    
  /* check the argument values and setup the filter, values, ... */
  rbv_init(nrhs,prhs);
  
  rbv_readData(nlhs,plhs);
  
  rbv_cleanup();
}

/*************************************************************
 *
 * a quick and dirty solution for the file size
 * I got it from http:/*bytes.com/forum/thread221370.html
 * They mention that it will work with binary data(which is fine for us)
 * but not with textual data.
 *
 * The function is only used, when HDR.nPoints was not set
 *
 * returns the length of a file (in bytes)
 *
 *************************************************************/
static int file_length(FILE *f)
{
  int pos;
  int end;

  pos = ftell (f);
  fseek (f, 0, SEEK_END);
  end = ftell (f);
  fseek (f, pos, SEEK_SET);

  return end;
}

/************************************************************
 *
 * Checks if the input values are valid and sets up the values
 * for reading the eeg-file
 *
 ************************************************************/

static void rbv_init(int nrhs, const mxArray *prhs[])
{
  const mxArray *FILE_NAME;
  const  mxArray *HDR;
  const mxArray *OPT;
  mxArray *tempPointer;
  bool isAFilter, isBFilter; /* to check if filt_a and filt_b was set  */
  int lag; /* the difference between the sampling rate of the raw data and
              and the sampling rate of the requested data */
  int i;  /* temp counting value  */
  double* tempDataPtr; /* pointer for OPT.dataPos */
  
  char *charBuf;            /* buffer for char reading (please free after usage)  */
  mwSize charBufLength;     /* the size of the buffer  */
  
  /*
   * opening the eeg file
   */
  FILE_NAME = prhs[0];
  charBufLength = mxGetNumberOfElements(FILE_NAME) + 1;
  charBuf = malloc(charBufLength * sizeof(char));
  
  if (mxGetString(FILE_NAME, charBuf, charBufLength) != 0) {
    free(charBuf); charBuf = 0;
    mexErrMsgTxt("Could not read file name.");
    return;
  }
  eegFile = fopen(charBuf,"rb");  /* only r will cause an error rb stands  */
                                  /* for read binary  */
  rbv_assert(NULL != eegFile, "Could not open eeg file.");
  free(charBuf); charBuf = 0;

  /*
   * HDR loading
   */
  HDR = prhs[1];
  rbv_assert(mxIsStruct(HDR),"HDR has to be a struct.");
  /* see if we have our fields */
  rbv_assert(mxGetFieldNumber(HDR,FS_FIELD) != -1,"The field HDR.fs was not set");
  rbv_assert(mxGetFieldNumber(HDR,N_CHANS_FIELD) != -1,"The field HDR.nChans was not set");
  rbv_assert(mxGetFieldNumber(HDR,SCALE_FIELD) != -1,"The field HDR.scale was not set");
  rbv_assert(mxGetFieldNumber(HDR,ENDIAN_FIELD) != -1,"The field HDR.endian was not set");
 
  /* 
   * load the field HDR.fs 
   */
  tempPointer = mxGetField(HDR,0,FS_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "HDR.fs must be a real scalar.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "HDR.fs argument must be real scalar.");
  rawDataSamplingRate = mxGetScalar(tempPointer);
  
  /*
   * load the field HDR.nChans 
   */
  tempPointer = mxGetField(HDR,0,N_CHANS_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "HDR.nChans must be a real scalar.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "HDR.nChans argument must be real scalar.");
  rawDataChannelCount = (int)mxGetScalar(tempPointer);
  
  /*
   * load the field HDR.nPoints 
   */
  if(mxGetFieldNumber(HDR,N_POINTS_FIELD) != -1) {
    /* nPoints was set  */
    tempPointer = mxGetField(HDR,0,N_POINTS_FIELD);
    rbv_assert(mxIsNumeric(tempPointer), "HDR.nPoints must be a real scalar.");
    rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
       "HDR.nPoints argument must be real scalar.");
    rawDataPoints = (int)mxGetScalar(tempPointer);
  } else {
    /* was not set  */
    rawDataPoints = file_length(eegFile) / (INT_16_SIZE * rawDataChannelCount);  
  }
  
  /* 
   * load the field HDR.scal 
   */
  tempPointer = mxGetField(HDR,0,SCALE_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "HDR.scale must be a real scalar vector.");
  rbv_assert(mxGetM(tempPointer) == 1, "HDR.scale has to be a vector.");
  rbv_assert(mxGetN(tempPointer) == rawDataChannelCount, 
        "HDR.scale has to be a vector with the size of nChanns.");
  
  rawDataScale = mxGetPr(tempPointer);
  
  /* 
   * load the field HDR.endian 
   */
  tempPointer = mxGetField(HDR,0,ENDIAN_FIELD);
  rbv_assert(mxIsChar(tempPointer), "HDR.endian must be a character.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "HDR.endian argument must be character.");
  charBufLength = mxGetNumberOfElements(tempPointer) + 1;
  charBuf = malloc(charBufLength * sizeof(char));
  
  if (mxGetString(tempPointer, charBuf, charBufLength) != 0) {
    free(charBuf); charBuf = 0;
    mexErrMsgTxt("Could not read HDR.endian .");
    return;
  }
  rawDataEndian = charBuf[0];
  free(charBuf); charBuf = 0;
  rbv_assert((rawDataEndian == 'l') || (rawDataEndian == 'b'),
     "HDR.endian must be 'l' or 'b', see documentation.");
  
  /*
   * OPT loading
   */
  OPT = prhs[2];
  rbv_assert(mxIsStruct(OPT),"OPT has to be a struct.");
  /* see if we have our fields */
  rbv_assert(mxGetFieldNumber(OPT,CHAN_ID_X_FIELD) != -1,"The field OPT.chanidx was not set");
  rbv_assert(mxGetFieldNumber(OPT,FS_FIELD) != -1,"The field OPT.fs was not set");
  /* the others are optional a will be dealt with in creation  
  
  /* 
   *load the field OPT.chanidx 
   */
  tempPointer = mxGetField(OPT,0,CHAN_ID_X_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "OPT.chanidx must be a real scalar vector.");
  rbv_assert(mxGetM(tempPointer) == 1, "OPT.chanidx has to be a vector.");
  optChannelSelect = mxGetPr(tempPointer);
  optChannelSelectCount = mxGetN(tempPointer);
  
  /* 
   *load the field OPT.fs 
   */
  tempPointer = mxGetField(OPT,0,FS_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "OPT.fs must be a real scalar.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "OPT.fs argument must be real scalar.");
  optSamplingRate = mxGetScalar(tempPointer);
  
  /* calculate lag see the value creation for more details */
  lag = (int) (rawDataSamplingRate / optSamplingRate);
  
  /*
   * load the data field and dataPos field
   */
  if(mxGetFieldNumber(OPT,DATA) != -1) {
    tempPointer = mxGetField(OPT,0,DATA);
    rbv_assert(mxIsNumeric(tempPointer), "OPT.data must be a real scalar matrix.");
    rbv_assert(mxGetN(tempPointer) == optChannelSelectCount,
        "OPT.data must have the same size as chanidx.");
    dataStart = 0;
    dataPtrSize = dataEnd = mxGetM(tempPointer);
    
    /* these two are set wiht dataPos or in read data */
    fileStart = -1;
    fileEnd = -1;
    
    dataPtr = mxGetPr(tempPointer); 
    
    if(mxGetFieldNumber(OPT,DATA_POS) != -1) {
      tempPointer = mxGetField(OPT,0,DATA_POS);
      rbv_assert(mxIsNumeric(tempPointer), "OPT.dataPos must be a real scalar vector.");
      rbv_assert(mxGetM(tempPointer) == 1,
          "OPT.dataPos must be a vector.");
      rbv_assert(mxGetN(tempPointer) == 4, "OPT.dataPos must have the size 4");
      
      tempDataPtr = mxGetPr(tempPointer);
      dataStart = (int)tempDataPtr[0];
      dataEnd = (int)tempDataPtr[1];
      fileStart = (int)tempDataPtr[2];
      fileEnd = (int)tempDataPtr[3];
      
    }
  }
  
  
  /* 
   * load the IIR filter if it was set 
   */
  isAFilter = mxGetFieldNumber(OPT,FILT_A_FIELD) != -1;
  isBFilter = mxGetFieldNumber(OPT,FILT_B_FIELD) != -1;
  
  if(isAFilter && isBFilter) {
    /* load the filters from the structure */
    tempPointer = mxGetField(OPT,0,FILT_A_FIELD);
    rbv_assert(mxIsNumeric(tempPointer), "OPT.filt_a must be a real scalar vector.");
    rbv_assert(mxGetM(tempPointer) == 1, "OPT.filt_a has to be a vector.");
    iirFilterSize = mxGetN(tempPointer);
    
    aFilter = malloc(iirFilterSize * sizeof(double));
    memcpy(aFilter, mxGetPr(tempPointer), iirFilterSize*sizeof(double));
    
    tempPointer = mxGetField(OPT,0,FILT_B_FIELD);
    rbv_assert(mxIsNumeric(tempPointer), "OPT.filt_b must be a real scalar vector.");
    rbv_assert(mxGetM(tempPointer) == 1, "OPT.filt_b has to be a vector.");
    
    bFilter = malloc(iirFilterSize * sizeof(double));
    memcpy(bFilter, mxGetPr(tempPointer), iirFilterSize*sizeof(double));
    
    rbv_assert(iirFilterSize == mxGetN(tempPointer), 
        "OPT.filt_a and OPT.filt_b must have the same size.");
  } else {
    if(isAFilter == isBFilter) {
      /* both are false, we will create a default one  */
      iirFilterSize = 1;
      
      bFilter = malloc(iirFilterSize * sizeof(double));
      aFilter = malloc(iirFilterSize * sizeof(double));
      
      bFilter[0] = 1.0;
      aFilter[0] = 1.0;
    } else {
       mexErrMsgTxt("OPT.filt_a or OPT.filt_b was not set.");
       return;
    }
  }
  
  /* 
   * load the FIR filter if it was set 
   */
  if(mxGetFieldNumber(OPT,FILT_SUBSAMPLE_FIELD) != -1) {
    /* load filter from the structure */
    tempPointer = mxGetField(OPT,0,FILT_SUBSAMPLE_FIELD);
    rbv_assert(mxIsNumeric(tempPointer), "OPT.filt_subsample must be a real scalar vector.");
    rbv_assert(mxGetM(tempPointer) == 1, "OPT.filt_subsample has to be a vector.");
    firFilterSize = mxGetN(tempPointer);
    
    rbv_assert(firFilterSize == lag,"FIR filter has to correspondent with the sampling rate.");
    firFilter = malloc(firFilterSize * sizeof(double));
    memcpy(firFilter, mxGetPr(tempPointer), firFilterSize*sizeof(double));
  } else {
    /* the defalut filter will only take the last value from each block  */
    firFilterSize = lag;
    firFilter = malloc(firFilterSize * sizeof(double));

      
    for(i = 0; i < firFilterSize;++i) {
      firFilter[i] = 0.0;
    }
    firFilter[firFilterSize - 1] = 1.0;
  }
  
  /* create ant initialize the buffers for the IIR filter */
  xBuffer = malloc(iirFilterSize * optChannelSelectCount * sizeof(double));
  yBuffer = malloc(iirFilterSize * optChannelSelectCount * sizeof(double));

  for(i = 0; i < optChannelSelectCount * iirFilterSize;++i) {
    xBuffer[i] = 0.0;
    yBuffer[i] = 0.0;
  }
  
  /* create and initialize the buffers for the FIR filter */
  firFilterPosition = 0;
  firFilterSum = 0.0;    
  for(i = 0; i < firFilterSize;++i) {
    firFilterSum += firFilter[i];
  }

  /* when the FIR filter has the form [0 ... 0 ] the sum will be 0  */
  /* and we will get a divide by sero exeption  */
  if(firFilterSum == 0) {
    firFilterSum = 1.0;
  }

  firFilterValues = malloc(optChannelSelectCount * sizeof(double));
  for(i = 0; i < optChannelSelectCount;++i) {
    firFilterValues[i] = 0;
  }
}

/*************************************************************
 *
 * reads a data block from the eeg-file we assume, that the
 * eeg-file has following properties:
 *  DataFormat = BINARY
 *  DataOrientation = MULTIPLEXED
 *  BinaryFormat = INT_16
 *
 * With swap we will determine if the endianes of this machine is different
 * to the endianes of the data.
 *
 *
 * If we have an eeg-file with multiplexed data layout, the data is
 * packet channel wise:
 *
 * |Value1:Chan1|Value1:Chan2| ... |Value1:ChanX|Value2:Chan1|Value2:Chan2| ...
 * |ValueY:Chan1| ... |ValueY:ChanX|EOF
 *
 *************************************************************/
static bool 
rbv_readDataBlock(double *dataBlock, short* dataBuffer, int channelCount ,bool swap)
{
  /* We need to check this code on 64-bit machines it might not work.  */
  int dataRead;
  int i;
  char temp;
  char *tempShortPointer;
  
  dataRead = fread(dataBuffer, INT_16_SIZE, channelCount, eegFile);
  if(dataRead == 0) {
    return false;
  } else {
    for(i = 0; i < channelCount;++i) {
      if(swap) {
        /* swap the first and second byte  */
		    tempShortPointer = (char*)&dataBuffer[i];
		    temp = tempShortPointer[0];
        tempShortPointer[0] = tempShortPointer[1];
        tempShortPointer[1] = temp;      
      }

      dataBlock[i] = (double)dataBuffer[i];
    }
  }
  
  return true;
}

/*************************************************************
 *
 * Checks the endian format of this machine
 *
 *************************************************************/
static char endian() {
    int i = 1;
    char *p = (char *)&i;

    if (p[0] == 1)
        return 'l'; /*least important byte is first byte  */
    else
        return 'b'; /*least important byte is last byte  */
}
  

/************************************************************
 *
 * Get Data from the file
 *
 ************************************************************/

static void 
rbv_readData(int nlhs, mxArray *plhs[])
{  
  double *outData;          /* the return data of the matlab matrix as double array  */
  int outDataPos;           /* the number of data blocks written to the outData  */
  int rawDataPos;			      /* the number of data blocks read from the file  */
  int outDataSize;          /* the number of blocks in the outdata  */
  bool swap;                /* swap the bytes of the data */
  
  double *dataBlock;        /* one data block we will read from the file */
  short  *readBuffer;       /* a temporary array we will actually read to */
  double selectedChannelValue; /* one value from the eeg data */
  int selectedChannel, n;
  
  swap = rawDataEndian != endian();
  dataBlock = malloc(rawDataChannelCount * sizeof(double));
  readBuffer = malloc(rawDataChannelCount * sizeof(short));
  
  /* construct the data output matrix. */
  outDataSize = rawDataPoints / firFilterSize; 
  
  if(fileStart == -1) {
    fileStart = 0;
  }
  if(fileEnd == -1) {
    fileEnd = outDataSize;
  }
  
  if(dataPtr == 0) {
    plhs[0] = mxCreateDoubleMatrix(outDataSize, optChannelSelectCount, mxREAL);
    outData = mxGetPr(plhs[0]);
  } else {
    outData = dataPtr;
    outDataSize = dataPtrSize;
  }
  outDataPos = 0;
  
  
  for(rawDataPos = 0; rawDataPos < rawDataPoints; ++rawDataPos) {
   rbv_readDataBlock(dataBlock,readBuffer, rawDataChannelCount,swap);
   for(n = 0; n < optChannelSelectCount;++n) {
     selectedChannel = (int)optChannelSelect[n];
     selectedChannelValue = dataBlock[selectedChannel]; /* get the value */
     selectedChannelValue *= rawDataScale[selectedChannel]; /* scal the value */
     
     selectedChannelValue = rbv_filterDataIIR(selectedChannelValue, n); /* IIR Filter */
     firFilterValues[n] += selectedChannelValue; /* fir Filter */
   }
   
   /* update the count of the values in the fir filter */
   firFilterPosition++;

   /* flush the fir filter and write to outData */
   if(firFilterPosition == firFilterSize) {
     firFilterPosition = 0;

     for(n = 0;n < optChannelSelectCount; ++n) {
       if(fileStart <= outDataPos && 
          outDataPos <= fileEnd &&  /*we only set the data when we are in the range*/
          dataEnd >= outDataPos + dataStart - fileStart) { /* check for the bounds of the data*/
        outData[n * outDataSize + outDataPos + dataStart - fileStart] = firFilterValues[n] / firFilterSum;
       }
       firFilterValues[n] = 0;
     }
     outDataPos++;
  }

   /* update the postition of the latest data in the IIR filter */
   --iirFilterOffset;
   if( iirFilterOffset < 0) {
     iirFilterOffset = iirFilterSize - 1;
   }
  }
    
  free(dataBlock);
  free(readBuffer);
}

/************************************************************
 *
 * Calculates the value from the IIR filter and updates the 
 * filter for the next value in the same channel
 *
 * The data in both buffer is packed like;
 *
 *  1. Channel data0       data1       data2       ... dataK
 *  2. Channel datak+1     datak+2     datak+3     ... dataK*2
 *  .
 *  . 
 *  .
 *  n. Channel data(n-1)+1 data(n-1)+2 data(n-1)+3 ... data(n-1)*K
 *
 * where K is the filterSize-1
 * and n is the nummber of channels
 *
 * The latest dataPoint for each channel is at the position
 * iirFilterOffset. The oldest dataPoint for each channel is at the
 * position iirFilterOffset - 1.
 *
 ************************************************************/
static double rbv_filterDataIIR(double value, int channel) {
  int channelOffset;
  int bufferPos;
  double yValue;
  int i;
  
  /* updates the filter in assumtion of aFilter[0] = 1
   * the expression for an IIR filter is
   * a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
   *                       - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
   * see matlab: help filter
   * for more information
   */
  
  yValue = 0.0;
  channelOffset = channel * iirFilterSize;
  xBuffer[iirFilterOffset + channelOffset] = value; /* set x(n)  */
  yBuffer[iirFilterOffset + channelOffset] = 0.0;   /* set y(n) for the calculation  */
  
  for(i = 0; i < iirFilterSize; ++i) {
    bufferPos = (i + iirFilterOffset) % iirFilterSize + channelOffset;
    yValue = yValue + xBuffer[bufferPos] * bFilter[i] - yBuffer[bufferPos] * aFilter[i];
  }
  
  yBuffer[iirFilterOffset + channelOffset] = yValue; /* set y(n) the filtered value  */
  
  return yValue;
}

/************************************************************
 *
 * Free the space we have used and close the eeg file.
 *
 ************************************************************/

static void rbv_cleanup()
{
  if(0 != eegFile) {
    fclose(eegFile);
    eegFile = 0;
  }
  if(0 != xBuffer) {
    free(xBuffer); 
    xBuffer = 0;
  }
  if(0 != yBuffer) {
    free(yBuffer); 
    yBuffer = 0;
  }
  if(0 != aFilter) {
    free(aFilter); 
    aFilter = 0;
  }
  if(0 != bFilter) {
    free(bFilter); 
    bFilter = 0;
  }
  if(0 != firFilter) {
    free(firFilter); 
    firFilter = 0;
  }
  if(0 != firFilterValues) {
    free(firFilterValues); 
    firFilterValues = 0;
  }
}

/************************************************************
 *
 * checks for errors and does some cleanup before returning to matlab
 *
 ************************************************************/
static void rbv_assert(bool aValue,const char *text) {
  if(!aValue) {
    rbv_cleanup();
    
    mexErrMsgTxt(text);
  }
}
