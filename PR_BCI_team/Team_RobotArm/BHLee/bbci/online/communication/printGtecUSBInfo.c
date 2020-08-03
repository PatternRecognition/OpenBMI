/*
 * printGtecUSBInfo.c
 *
 * This mex file searchs for all connect usb amps and prints the usb port,
 * serial number, driver version and the hardware version.
 *
 * 2010/02/16 - Max Sagebaum
 *               - file created 
 *
 * (c) Fraunhofer FIRST.IDA 2010
 */

#include <windows.h>
#include <math.h>

/* The includes for the gTec controllers */
#include "gUSBamp.h"
#pragma comment(lib,"gUSBamp.lib")

#include "mex.h"

/* placeholders for the functions */
static int printInfo();
static void printError();

/*********************************************
 *
 * The entry point for matlab.
 * 
 *********************************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  printInfo();
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
static int printInfo(){
  HANDLE hdev;
  int i, contrPos;
  float driverVersion;
  float hardwareVersion;
  LPSTR* serialNumber;
  
  serialNumber = malloc(20*sizeof(LPSTR));
  
  driverVersion = GT_GetDriverVersion();
  mexPrintf("Driver Version: %f\n",driverVersion);
  
  /* find the divices connected to the pc */
  for(i = 1; i < 32; ++i) {
    hdev = GT_OpenDevice(i);
    
    if(NULL != hdev) {
      // found a divice
      FILT *filterSpecs;
      FILT *notchFilterSpecs;
      int nFilterSpecs;
      int nNotchFilterSpecs;
      int n;

      hardwareVersion = GT_GetHWVersion(hdev);
      if(0 == GT_GetSerial(hdev, serialNumber,20)) {
        printError();
        return 0;
      }
      mexPrintf("USB-Port: %i hardwareVersion: %f SerialNumber: %s\n",i, hardwareVersion, serialNumber);
      
      if(0 == GT_GetNumberOfFilter(&nFilterSpecs)) {
        printError();
        return 0;
      }
      
      if(0 == GT_GetNumberOfNotch(&nNotchFilterSpecs)) {
        printError();
        return 0;
      }
      
      filterSpecs = (FILT*) malloc(nFilterSpecs * sizeof(FILT));
      notchFilterSpecs = (FILT*) malloc(nNotchFilterSpecs * sizeof(FILT));      
      
      if(0 == GT_GetFilterSpec(filterSpecs)) {
        printError();
        return 0;
      }
      
      if(0 == GT_GetNotchSpec(notchFilterSpecs)) {
        printError();
        return 0;
      }
      
      for(n=0;n<nFilterSpecs;++n) {
        mexPrintf("  Filter %i: fu: %f fo: %f fs: %f type: %f order: %f \n",
          n,
          filterSpecs[n].fu,
          filterSpecs[n].fo,
          filterSpecs[n].fs,
          filterSpecs[n].type,
          filterSpecs[n].order);
      }
      
      for(n=0;n<nNotchFilterSpecs;++n) {
        mexPrintf("  Notch %i: fu: %f fo: %f fs: %f type: %f order: %f \n",
          n,
          notchFilterSpecs[n].fu,
          notchFilterSpecs[n].fo,
          notchFilterSpecs[n].fs,
          notchFilterSpecs[n].type,
          notchFilterSpecs[n].order);
      }
      
      
      GT_CloseDevice(&hdev);
    } 
  }
  
  return 1;
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

