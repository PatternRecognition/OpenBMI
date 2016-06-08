#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include "mex.h"
/*
 * This file loads the library inpOut*.dll an extracts the functions for the 
 * parallelport communication.
 *
 * The libraries can be found at http://www.highrez.co.uk/Downloads/InpOut32/default.htm
 *
 * If you have problems with the parallelport especial under windows 7 and vista
 * do the following:
 * Run matlab once with administrator rights. Then the drivers
 * will be installed in the system and you can use them without the 
 * administrator rights.
 *
 * 2011/01/25 - Max Sagebaum
 *              - added docs
 *              - added test if the internal library routines could be loaded.
*/

const char *TRIGGER_FIELD = "Trigger";
const char *IO_LIB_FIELD = "IO_LIB";


/* prototype (function typedef) for DLL function Inp32: */
    typedef void (_stdcall *oupfuncPtr)(short portaddr, short datum);
    typedef short (_stdcall *inpfuncPtr)(short portaddr);
    typedef BOOL	(_stdcall *isInpOutDriverOpenPtr)();
     
 /* After successful initialization, these 2 variables
   will contain function pointers.
 */
     oupfuncPtr oup32fp;
     inpfuncPtr inp32fp;
     isInpOutDriverOpenPtr isOpenFP;
     
     
 /* Wrapper functions for the function pointers
    - call these functions to perform I/O.
 */
     

     void  Out32 (short portaddr, short datum)
     {
          (oup32fp)(portaddr,datum);
     } 
     
     short  Inp32 (short portaddr)
     {
          return (inp32fp)(portaddr);
     }
     
  /* 
   creation of the lebrary
  */
     HINSTANCE createLibrary() {
         char* buf;
         int buflen;
         HINSTANCE hLib;
         mxArray *var_BBCI;
         
         var_BBCI = mexGetVariable("global", "BBCI");
         if(array_ptr == NULL) {
             mexErrMsgTxt("IO_LIB not found or is nor global variable.");
         }
         rbv_assert(mxIsStruct(var_BBCI),"global BBCI has to be a struct.");
         /* see if we have our fields */
         rbv_assert(mxGetFieldNumber(var_BBCI,TRIGGER_FIELD) != -1,"The field BBCI.Trigger was not set");
         var_Trigger = mxGetField(var_BBCI,0,TRIGGER_FIELD);
         rbv_assert(mxIsStruct(var_Trigger),"BBCI.Trigger has to be a struct.");
         rbv_assert(mxGetFieldNumber(var_Trigger,IO_LIB_FIELD) != -1,"The field BBCI.Trigger.IO_LIB was not set");
         var_IO_LIB = mxGetField(var_Trigger,0,IO_LIB_FIELD);
         
         buflen = mxGetNumberOfElements(var_IO_LIB) + 1;
         buf = mxCalloc(buflen, sizeof(char));
  
         mxGetString(var_IO_LIB, buf, buflen);
         
         hLib = LoadLibrary(buf);
         //hLib = LoadLibrary("inpout32.dll");

         if (hLib == NULL) {
              mexErrMsgTxt("LoadLibrary Failed. Check IO_LIB.\n");
         }

         /* get the address of the function */

         oup32fp = (oupfuncPtr) GetProcAddress(hLib, "Out32");

         if (oup32fp == NULL) {
              mexErrMsgTxt("GetProcAddress for Oup32 Failed.\n");
         }
         
         inp32fp = (inpfuncPtr) GetProcAddress(hLib, "Inp32");

         if (inp32fp == NULL) {
              mexErrMsgTxt("GetProcAddress for Inp32 Failed.\n");
         }
         
         isOpenFP = (isInpOutDriverOpenPtr) GetProcAddress(hLib, "IsInpOutDriverOpen");
         
         if (isOpenFP == NULL) {
              mexErrMsgTxt("GetProcAddress for IsOpen Failed.\n");
         }
         
         if(!isOpenFP()) {
             mexErrMsgTxt("Parellel Port library is not loaded.(Try running as an administrator).\n");
         }
         
         return hLib;
    }
    
    void deleteLibrary(HINSTANCE hLib) {
        FreeLibrary(hLib);
    }
        
