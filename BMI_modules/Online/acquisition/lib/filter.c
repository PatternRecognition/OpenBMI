/*
 * filter.c
 *
 * This file contains the logic for the IIR and FIR filters used by acquire_bv
 * and acquire_gtec.
 *
 * 2009/01/09 - Max Sagebaum
 *              - file created
 *              - the logic in this file is form acquire_bv. The logic was
 *                moved to reuse it in acquire_gtec
 *			
 * 2009/08/24 - Max Sagebaum
 *			    - logic for the IIR filter changed
 * 2009/09/18 - Max Sagebaum 
 *          - introduced a local channel for the IIR logic
 *          - fixed a possible bug: If the channel selection was changed 
 *              you got an invalid memory access exception
 * 2010/08/26 - Max Sagebaum
 *              - Added a function to get the filter size of the fir filter.
 */

#include "filter.h"

/* the values for the IIR filter */
static int filterSize = 0;            /* the size of the IIR filter */
static int channelCount;              /* the number of bci channels */
static double *bFilter;               /* the b part of the IIR filter */
static double *aFilter;               /* the a part of the IIR filter */

static double *zBuffer;               /* the internal buffer for the filter, one buffer for each row */


/* the values for the resampling of the data */
static double *reSampleFilter;        /* a filter for the resampling of the data */
static int    reSampleFilterPosition; /* the position in the filter */
static int    reSampleFilterSize;     /* the size of the filter */
static double *reSampleFilterValues;  /* the current values for the resampling */

/************************************************************
 *
 * Creates the FIR filter and sets all reSampleFilter* values.
 * INPUT: filter   - the values for the FIR filter
 *        size     - The size of the FIR filter
 *        nChans   - The number of channels we have to filter
 *
 ************************************************************/
static void filterFIRCreate(double* filter, int size, int nChans) {
  int i;
  
  reSampleFilterSize = size;
  reSampleFilterPosition = 0;

  reSampleFilter = (double*)malloc(reSampleFilterSize * sizeof(double));
  memcpy(reSampleFilter,filter,reSampleFilterSize * sizeof(double));
  
  reSampleFilterValues = (double*)malloc(nChans * sizeof(double));
  for(i = 0; i < nChans;++i) {
    reSampleFilterValues[i] = 0;
  } 
}

/************************************************************
 *
 * Creates the IIR filter values
 * INPUT: aFilterPtr   - The values for the a component of the filter
 *        bFilterPtr   - The values for the b component of the filter
 *        fSize        - The size of the IIR filter
 *        nChans       - The number of channels we have to filter
 *
 *  If you want to create a default filter with a = [1] and b = [1] you
 *  can pass for aFilterPtr and bFilterPtr NULL. The function assumes in
 *  this situation that fSize is equal to one.
 *
 ************************************************************/
static void filterIIRCreate(double* aFilterPtr, double* bFilterPtr,int fSize, int nChans) {
  int i;
  
  filterSize = fSize;
  channelCount = nChans;
  
  bFilter = (double*)malloc(filterSize * sizeof(double));
  aFilter = (double*)malloc(filterSize * sizeof(double));
  
  if(NULL != aFilterPtr) {
    memcpy(bFilter, bFilterPtr, filterSize*sizeof(double));
    memcpy(aFilter, aFilterPtr, filterSize*sizeof(double));
  } else {
    /* if the a filter is the null pointer we have to initialize the default
     * filter.
     *
     * size is assumend to be 1
     */
    
     bFilter[0] = 1.0;
     aFilter[0] = 1.0;
  }
  
  
  zBuffer = (double*)malloc(filterSize * channelCount * sizeof(double));
  
  for(i = 0; i < channelCount * filterSize;++i) {
    zBuffer[i] = 0.0;
  }
}

/************************************************************
 *
 * Calculates the value from the IIR filter and updates the 
 * filter for the next value in the same channel
 *
 * The data in the both buffer is packed liked;
 *
 *  1. Channel data0       data1       data2       ... dataK
 *  2. Channel datak+1     datak+2     datak+3     ... dataK*2
 *  .
 *  . 
 *  .
 *  n. Channel data(n-1)+1 data(n-1)+2 data(n-1)+3 ... data(n-1)*K
 *
 * where K is the filterSiyze-1
 * and n is the nummber of channels
 *
 * The latest dataPoint for each channel is at the position
 * filterOffset. The oldest dataPoint for each channel is at the
 * position filterOffset - 1.
 *
 ************************************************************/
static double filterDataIIR(double value, int channel) {
  int channelOffset;
  double yValue;
  int i;
  
  double* zBufferThisChannel; /* The pointer to the z buffer for the current channel */
  
  /* updates the filter in assumtion of aFilter[0] = 1
   * the expression for an IIR filter is
   * a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
   *                       - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
   * see matlab: help filter for more information
   *
   * We do not need the extra case for the last element because zBuffer[filterSize - 1] will always be zero
   */
  

  channelOffset = channel * filterSize;
  zBufferThisChannel = zBuffer + channelOffset;
  
  yValue = bFilter[0] * value + zBuffer[channelOffset];  
  for(i = 1; i < filterSize; ++i) {
    zBufferThisChannel[i - 1] = bFilter[i] * value + zBufferThisChannel[i] - aFilter[i] * yValue;    
  }

  return yValue;
}

/************************************************************
 *
 * Deletes all values for the two filters
 * 
 ************************************************************/
static void filterClose() {
  if(NULL != zBuffer) {free(zBuffer); zBuffer = NULL;}
  if(NULL != aFilter) {free(aFilter); aFilter = NULL;}
  if(NULL != bFilter) {free(bFilter); bFilter = NULL;}
  if(NULL != reSampleFilter) {free(reSampleFilter); reSampleFilter = NULL;}
  if(NULL != reSampleFilterValues) {free(reSampleFilterValues); reSampleFilterValues = NULL;}
}

/************************************************************
 *
 * Filters the data in the arrays with the filters. 
 * nChans is the value used in the init methods.
 * INPUT: sourceData      - The unfiltered data points
 *                          We assume that the size of the array is
 *                          sourceDataSize * nChans
 *        sourceDataSize  - The number of data sets in the array
 *        filterData      - The array for the return data
 *                          We assume that the size of the array is 
 *                          filterDataSize * chanl_selSize
 *        filterDataSize  - The number of data sets in the array
 *        chanl_sel       - The rearangement of channels
 *        chanl_selSize   - The size of the channel selection array
 *        scale           - The scale for the cahnnels
 *
 ************************************************************/
static void filterData(double* sourceData, int sourceDataSize, double* filterData, int filterDataSize,double* chan_sel, int chan_selSize, double* scale) {
  int t;
  int n;
  int c;
  int pDstPosition;
  double* pSrc;
  double* pDst;
  
  pSrc = sourceData;
  pDstPosition = 0;
  
  /* update thr IIR filter with the new values
     Copy the data (re-arranging
     the channels according to chan_sel, scaling the values
     according to scale) */
  for(t = 0; t < sourceDataSize; ++t) {
    /* IIR filter and resample filter  */
    for(n = 0;n < channelCount; ++n) {
     reSampleFilterValues[n] += filterDataIIR(pSrc[n],n) * reSampleFilter[reSampleFilterPosition];
    }
    reSampleFilterPosition++;

    /* flush the resample filter and write to dest  */
    if(reSampleFilterPosition == reSampleFilterSize) {
      reSampleFilterPosition = 0;

      /* write to dest */
      pDst = filterData + pDstPosition;
      for(n = 0; n < chan_selSize; ++n) {
        c = (int)chan_sel[n] - 1; /* we have matlab indices here so we need to substract one */
        *pDst = scale[c] * reSampleFilterValues[c];
        pDst+= filterDataSize;
      }
      
      /* flush the data */
      for(n = 0;n < channelCount; ++n) {        
        reSampleFilterValues[n] = 0;
      }
      pDstPosition++;
    }

    pSrc += channelCount;
  }
}

/************************************************************
 *
 * Get the position of the FIR filter. You can use it to determine
 * how big the filterData has to be for the filterData method.
 *
 ************************************************************/
static int getFIRPos() {
  return reSampleFilterPosition;
}

/************************************************************
 *
 * Set the values of the FIR filter. We assusme that the array has the same
 * size as the initial FIR filter.
 *
 ************************************************************/
static void filterFIRSet(double* filter) {
  memcpy(reSampleFilter,filter,reSampleFilterSize * sizeof(double));
}

static int filterGetFIRSize() {
  return reSampleFilterSize;
}
