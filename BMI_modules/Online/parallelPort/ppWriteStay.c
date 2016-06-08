#include <stdio.h>
#include "mex.h"
#include <windows.h>
#include <winbase.h>
#include "ioPort.c"

#define pAddr prhs[0]
#define pData prhs[1]

/* ppWriteStay(IO_ADDR, Value) is used like ppWrite. This function will not 
   set the parallel port value to 0 after 10 ms. It has to be done by the user
   itself.
 
   If you have problems with the parallelport see ioPort.c. 
 
   Max Sagebaum         02.11.2007
*/

typedef unsigned short ioaddr_t;
typedef unsigned char portvalue_t;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  ioaddr_t ioaddr;
  portvalue_t portvalue;
  HINSTANCE hLib;
  
  hLib = createLibrary();
  
  if (nrhs!=2)
	mexErrMsgTxt("Three input arguments required");
  if (nlhs>0)
	mexErrMsgTxt("No output argument allowed");

  /* get parameters from mex format and store into struct p */
  ioaddr= (unsigned short)*mxGetPr(pAddr);
  portvalue= (unsigned char)*mxGetPr(pData);

  //mexPrintf("Intent to write %d, current value is %d\n", p->sendvalue, currentvalue); 
  /* ToDo: Put mutex lock around the following two lines */
  Out32(ioaddr, portvalue);
  
  deleteLibrary(hLib);
}
