/*
 * filter.h
 *
 * This is the header filer for filter.c. It contains the declarations for 
 * the functions in filter.h and hides the static variables in filter.c
 *
 * The file is currently used in acquire_bv.c, acquire_gtec.c and read_bv.c
 *
 * 2009/01/09 - Max Sagebaum
 *              - file created
 * 2009/08/24 - Max Sagebaum
 *              - usage update
 * 2009/09/18 - Max Sagebaum 
 *              - The signature of the method filterData was changed
 * 2010/08/26 - Max Sagebaum
 *              - Added a function to get the filter size of the fir filter.
 */

#ifndef FILTER_H
#define FILTER_H

#include "filter.c"

static void filterFIRCreate(double* filter, int size, int nChans);
static void filterIIRCreate(double* aFilterPtr, double* bFilterPtr,int fSize, int nChans);
static void filterClose();
static double filterDataIIR(double value, int channel);
static void filterData(double* sourceData, int sourceDataSize, double* filterData, int filterDataSize,double* chan_sel, int chan_selSize, double* scale);
static int getFIRPos();
static void filterFIRSet(double* filter);
static int filterGetFIRSize();

#endif