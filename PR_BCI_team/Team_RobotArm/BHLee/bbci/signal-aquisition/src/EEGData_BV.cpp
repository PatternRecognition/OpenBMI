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
 
  (c) Fraunhofer FIRST.IDA 2008
*/

/* define int16_t, int32_t, etc. */



#include "EEGData_BV.h"

#include <boost/cstdint.hpp>
using namespace boost;


//#include "../../online/communication/filter.h"

/* the field names for HDR and OPT*/


/************************************************************
 *
 * mexFunction
 *
 ************************************************************/
//void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
//{
//  /* First check of argument counts and output */
//  rbv_assert(nrhs == 3, "Exactly three input arguments required.");
//  rbv_assert(nlhs <= 1, "One or two output arguments required.");
//    
//  /* check the argument values and setup the filter, values, ... */
//  rbv_init(nrhs,prhs);
//  
//  rbv_readData(nlhs,plhs);
//  
//  rbv_cleanup();
//}

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
int EEGData_BV::file_length(FILE *f)
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

void EEGData_BV::rbv_init(int nrhs, string filename )
{

	/**FORMAT_FIELD = 'BinaryFormat';
	*FS_FIELD = 'fs';
	*N_CHANS_FIELD = 'nChans';
	*N_POINTS_FIELD = 'nPoints';
	*SCALE_FIELD = 'scale';
	*ENDIAN_FIELD = 'endian';
	*CHAN_ID_X_FIELD = 'chanidx';
	*FILT_A_FIELD = 'filt_a';
	*FILT_B_FIELD = 'filt_b';
	*FILT_SUBSAMPLE_FIELD = 'filt_subsample';
	*DATA = 'data';
	*DATA_POS = 'dataPos';*/

  
  const  char *HDR;
  const char *OPT;
  char *tempPointer;
  char *aFilter;
  char *bFilter;
  bool isAFilter, isBFilter; /* to check if filt_a and filt_b was set  */
  
  int i;  /* temp counting value  */
  double* tempDataPtr; /* pointer for OPT.dataPos */
  
  
  /*
   * opening the eeg file
   */
  eegFile = fopen(filename.c_str(),"rb");  /* only r will cause an error rb stands for read binary  */
  
  rbv_assert(NULL != eegFile, "Could not open eeg file.");
  
  /*
   * HDR loading
   */
  
  
  rawDataSamplingRate = 1;
  
  /*
   * load the field HDR.nChans 
   */
  rawDataChannelCount = 32;

  /*
   * load the field HDR.BinaryFormat
   */
  binaryDataFormat = 1;
  
  if (binaryDataFormat==1)
      rawElementSize = sizeof(short);
  else if (binaryDataFormat==2)
      rawElementSize = sizeof(int);
  else if (binaryDataFormat==3)
      rawElementSize = sizeof(float);    
  
  
  /*
   * load the field HDR.nPoints 
   */
    rawDataPoints = 32;
  
    /* was not set  */
    // if.. rawDataPoints = file_length(eegFile) / (rawElementSize * rawDataChannelCount);  
  
  
  /* 
   * load the field HDR.scal 
   */

  int rawDataScale = 1;
  
  /* 
   * load the field HDR.endian 
   */
  
  rawDataEndian = 1;
  
  //rbv_assert((rawDataEndian == 'l') || (rawDataEndian == 'b'), "HDR.endian must be 'l' or 'b', see documentation.");
  
 
  //fixme
  optChannelSelect = 1.;
  optChannelSelectCount = 1;
  optSamplingRate = 1;
  
  /* calculate lag see the value creation for more details */
  lag = (int) (rawDataSamplingRate / optSamplingRate);
  
    
  //fixme
  dataStart = 0;
  dataEnd = 5;
  dataPtrSize = dataEnd = 1;
  dataPtr = 1.; 
    /* these two are set wiht dataPos or in read data */
  fileStart = -1;
  fileEnd = -1;
       
    
  


}

/*************************************************************
 *
 * reads a data block from the eeg-file we assume, that the
 * eeg-file has following properties:
 *  DataFormat = BINARY
 *  DataOrientation = MULTIPLEXED
 *  BinaryFormat = INT_16, INT_32 or FLOAT_32
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

void EEGData_BV::swap16(char *b)
{
    char temp = b[0]; b[0]=b[1]; b[1]=temp;   
}

void EEGData_BV::swap32(char *b)
{
    char temp;
    temp = b[0]; b[0]=b[3]; b[3]=temp;
    temp = b[1]; b[1]=b[2]; b[1]=temp;    
}

bool EEGData_BV::rbv_readDataBlock(double *dataBlock, void *dataBuffer, int channelCount, bool swap)
{
  /* We need to check this code on 64-bit machines it might not work.  */
    int dataRead;
    int i;
	while (!feof(eegFile))
    dataRead = fread(dataBuffer, rawElementSize, channelCount, eegFile);
	
    if (dataRead == 0) {
        return false;
    } else {
        if (binaryDataFormat==1) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap16( (char*) &(((int16_t *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((int16_t *)dataBuffer)[i];
            }
        } else if (binaryDataFormat==2) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap32( (char*) &(((int32_t *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((int32_t *)dataBuffer)[i];
            }
        } else if (binaryDataFormat==3) {
            for (i = 0; i < channelCount; ++i) {
                if (swap) swap32( (char*) &(((float *)dataBuffer)[i]) );
                dataBlock[i] = (double) ((float *)dataBuffer)[i];
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
char EEGData_BV::endian() {
    int i = 1;
    char *p = (char *)&i;

    if (p[0] == 1)
        return 'l'; /*least important byte is first byte  */
    else
        return 'b'; /*least important byte is last byte  */
}
  

/************************************************************
 *
 * Free the space we have used and close the eeg file.
 *
 ************************************************************/

void EEGData_BV::rbv_cleanup()
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
void EEGData_BV::rbv_assert(bool aValue,const char *text) {
  if(!aValue) {
    rbv_cleanup();
    

  }
}
