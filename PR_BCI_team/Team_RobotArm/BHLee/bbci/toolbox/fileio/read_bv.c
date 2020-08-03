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
                - added option to load data into a matrix
  2008/07/17 - Max Sagebaum
				- checked for linux compatibly
  2008/09/20 - Marton Danoczy
                - reads ieee_float_32 brainvision files
  2009/08/24 - Max Sagebaum
				- updated IIR filter logic
  2009/09/18 - Max Sagebaum
                - now uses filter.h
  2009/11/29 - Marton Danoczy
                - reads int_32 brainvision files
                - included stdint.h to ensure data type lengths
  2010/09/09 - Max Sagebaum
                - added check that the opt.fs is a divisor of hdr.fs
  2012/02/09 - Benjamin Blankertz
                - added 64-Bit BinaryFormats
 
*/

/*define int16_t, int32_t, etc.*/
#ifdef _MSC_VER
#include "msvc_stdint.h"
#else
#include <stdint.h>
#endif

#include <string.h>
#include <stdio.h>
#include "mex.h"
#include "../../online/communication/filter.h"


/* the field names for HDR and OPT*/
const char *FORMAT_FIELD = "BinaryFormat";
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
static int rawDataSamplingRate;
static int rawDataChannelCount;
static int rawBinaryFormat;
static int rawElementSize;
static double *rawDataScale;
static char rawDataEndian;
static int rawDataPoints;

/* opt non optional values */
static double *optChannelSelect;
static int optChannelSelectCount;
static int optSamplingRate;

static int lag; /* the difference between the sampling rate of the raw data and
              and the sampling rate of the requested data */

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
rbv_readDataBlock(double *dataBlock, void* dataBuffer, int channelCount, bool swap);

static void rbv_readData( int nlhs, mxArray *plhs[]);

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
  mxArray *aFilter;
  mxArray *bFilter;
  bool isAFilter, isBFilter; /* to check if filt_a and filt_b was set  */
  
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
  rbv_assert(mxGetFieldNumber(HDR,FORMAT_FIELD) != -1,"The field HDR.BinaryFormat was not set");
  rbv_assert(mxGetFieldNumber(HDR,SCALE_FIELD) != -1,"The field HDR.scale was not set");
  rbv_assert(mxGetFieldNumber(HDR,ENDIAN_FIELD) != -1,"The field HDR.endian was not set");
 
  /* 
   * load the field HDR.fs 
   */
  tempPointer = mxGetField(HDR,0,FS_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "HDR.fs must be a real scalar.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "HDR.fs argument must be real scalar.");
  rawDataSamplingRate = (int)mxGetScalar(tempPointer);
  
  /*
   * load the field HDR.nChans 
   */
  tempPointer = mxGetField(HDR,0,N_CHANS_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "HDR.nChans must be a real scalar.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "HDR.nChans argument must be real scalar.");
  rawDataChannelCount = (int)mxGetScalar(tempPointer);

  /*
   * load the field HDR.BinaryFormat
   */
  tempPointer = mxGetField(HDR,0,FORMAT_FIELD);
  rbv_assert(mxIsNumeric(tempPointer), "HDR.BinaryFormat must be a real scalar.");
  rbv_assert(mxGetM(tempPointer) * mxGetN(tempPointer) == 1, 
	   "HDR.BinaryFormat argument must be real scalar.");
  rawBinaryFormat = (int)mxGetScalar(tempPointer);
  if (rawBinaryFormat==1)
      rawElementSize = sizeof(int16_t);
  else if (rawBinaryFormat==2)
      rawElementSize = sizeof(int32_t);
  else if (rawBinaryFormat==3)
      rawElementSize = sizeof(float);
  else if (rawBinaryFormat==4)
      rawElementSize = sizeof(double);
  else
      mexErrMsgTxt("Unknown Binary Format!");
  
  
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
    rawDataPoints = file_length(eegFile) / (rawElementSize * rawDataChannelCount);  
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
  optSamplingRate = (int)mxGetScalar(tempPointer);
  
  /* calculate lag see the value creation for more details */
  lag = (int) ((double)rawDataSamplingRate / (double)optSamplingRate);
  
  rbv_assert(lag * optSamplingRate == rawDataSamplingRate," The base frequency has to be a multiple of the requested frequency.");
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
    aFilter = mxGetField(OPT,0,FILT_A_FIELD);
    rbv_assert(mxIsNumeric(aFilter), "OPT.filt_a must be a real scalar vector.");
    rbv_assert(mxGetM(aFilter) == 1, "OPT.filt_a has to be a vector.");
    
	
    bFilter = mxGetField(OPT,0,FILT_B_FIELD);
    rbv_assert(mxIsNumeric(bFilter), "OPT.filt_b must be a real scalar vector.");
    rbv_assert(mxGetM(bFilter) == 1, "OPT.filt_b has to be a vector.");
    
    rbv_assert(mxGetN(aFilter) == mxGetN(bFilter), 
        "OPT.filt_a and OPT.filt_b must have the same size.");

    filterIIRCreate(mxGetPr(aFilter), mxGetPr(bFilter), mxGetN(aFilter), rawDataChannelCount);
  } else {
    if(isAFilter == isBFilter) {
      filterIIRCreate(NULL,NULL, 1, rawDataChannelCount);
    } else {
       mexErrMsgTxt("OPT.filt_a or OPT.filt_b was not set.");
       return;
    }
  }
  
  /* 
   * load the FIR filter if it was set 
   */
  if(mxGetFieldNumber(OPT,FILT_SUBSAMPLE_FIELD) != -1) {
    double* filter;

    /* load filter from the structure */
    tempPointer = mxGetField(OPT,0,FILT_SUBSAMPLE_FIELD);
    rbv_assert(mxIsNumeric(tempPointer), "OPT.filt_subsample must be a real scalar vector.");
    rbv_assert(mxGetM(tempPointer) == 1, "OPT.filt_subsample has to be a vector.");
    
    rbv_assert(mxGetN(tempPointer) == lag,"FIR filter has to correspondent with the sampling rate.");
    filter = malloc(lag * sizeof(double));
    memcpy(filter, mxGetPr(tempPointer), lag*sizeof(double));

    filterFIRCreate(filter, lag,rawDataChannelCount);
  } else {
    /* the defalut filter will only take the last value from each block  */
    double* filter;
    filter = malloc(lag * sizeof(double));

      
    for(i = 0; i < lag;++i) {
      filter[i] = 0.0;
    }
    filter[lag - 1] = 1.0;

    filterFIRCreate(filter, lag,rawDataChannelCount);
  }
}

/*************************************************************
 *
 * reads a data block from the eeg-file we assume, that the
 * eeg-file has following properties:
 *  DataFormat = BINARY
 *  DataOrientation = MULTIPLEXED
 *  BinaryFormat = INT_16, INT_32, FLOAT_32, or FLOAT_64
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

void swap16(char *b)
{
    char temp = b[0]; b[0]=b[1]; b[1]=temp;   
}

void swap32(char *b)
{
    char temp;
    temp = b[0]; b[0]=b[3]; b[3]=temp;
    temp = b[1]; b[1]=b[2]; b[1]=temp;    
}

static bool 
rbv_readDataBlock(double *dataBlock, void *dataBuffer, int channelCount, bool swap)
{
  /* We need to check this code on 64-bit machines it might not work.  */
    int dataRead;
    int i;
    
    dataRead = fread(dataBuffer, rawElementSize, channelCount, eegFile);
    if (dataRead == 0) {
        return false;
    } else {
        if (rawBinaryFormat==1) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap16( (char*) &(((int16_t *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((int16_t *)dataBuffer)[i];
            }
        } else if (rawBinaryFormat==2) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap32( (char*) &(((int32_t *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((int32_t *)dataBuffer)[i];
            }
        } else if (rawBinaryFormat==3) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap32( (char*) &(((float *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((float *)dataBuffer)[i];
            }
        } else if (rawBinaryFormat==4) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap32( (char*) &(((double *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((double *)dataBuffer)[i];
            }
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
  void  *readBuffer;       /* a temporary array we will actually read to */
  double *tempFilterData;   /* a buffer for the filtered data of one block */
  int n;
  
  swap = rawDataEndian != endian();
  dataBlock = malloc(rawDataChannelCount * sizeof(double));
  tempFilterData = malloc(optChannelSelectCount * sizeof(double));
  readBuffer = malloc(rawDataChannelCount * rawElementSize);
  
  /* construct the data output matrix. */
  outDataSize = rawDataPoints / lag; 
  
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

    filterData(dataBlock, 1, tempFilterData, 1 ,optChannelSelect, optChannelSelectCount, rawDataScale);
   
    /* check if the fir filter was flushed. Only if true some data was written to tempFilterData */
    if(0 == getFIRPos()) {

      if(fileStart <= outDataPos && 
        outDataPos <= fileEnd &&  /*we only set the data when we are in the range*/
        dataEnd >= outDataPos + dataStart - fileStart) { /* check for the bounds of the data*/
        for(n = 0;n < optChannelSelectCount; ++n) {
          outData[n * outDataSize + outDataPos + dataStart - fileStart] = tempFilterData[n];
        }
      }
      outDataPos++;
    }
  }
    
  free(dataBlock);
  free(readBuffer);
  free(tempFilterData);
}
/************************************************************
 *
 * Free the space we have used and close the eeg file.
 *
 ************************************************************/

static void rbv_cleanup()
{
  if(NULL != eegFile) {
    fclose(eegFile);
    eegFile = NULL;
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
