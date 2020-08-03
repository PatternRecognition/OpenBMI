#include <winsock.h>
#include <time.h>
#include "mex.h"
#include <windows.h>

#define pArgIn  prhs[0]
#define pArgOut plhs[0]
#define pArgOut2 plhs[1]



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  static clock_t lastTime;
  clock_t thisTime;
  double timeIncr, elaps;
  bool out;
  int rc;

  if (nrhs!=1)
    timeIncr= 0.0;
  else
    timeIncr= (double)*mxGetPr(pArgIn);
    
  if (timeIncr==0.0)
  { 
    rc= clock();
    if (rc<0)
    { mexErrMsgTxt("trouble with time function");
    }
    lastTime = rc;
    return;
  }

  rc= 0;
  thisTime= clock();
  elaps= (double)(thisTime-lastTime)/CLOCKS_PER_SEC*1000;
  
  out = (elaps>=timeIncr);

  if (out)
    lastTime = thisTime;
  
  if (nlhs>0)
  { pArgOut= mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(pArgOut)= out ? 1.0 : 0.0;
  }
  if (nlhs>1)
  { pArgOut2= mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(pArgOut2)= elaps-timeIncr;
  }
  
}

