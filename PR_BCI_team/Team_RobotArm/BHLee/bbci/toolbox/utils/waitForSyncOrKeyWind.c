/* Benjamin Blankertz */

#include <winsock.h>
#include <time.h>
#include <conio.h>
#include "mex.h"
#include <windows.h>

#define pArgIn  prhs[0]
#define pArgOut plhs[0]
#define pArgOut2 plhs[1]



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  static double lastTime, tickspermsec;
  double timeIncr, elaps, out;
  int toolate;
  long long pc, pcfreq;

  if (nrhs!=1)
    timeIncr= 0.0;
  else
    timeIncr= (double)*mxGetPr(pArgIn);
    
  if (timeIncr==0.0)
  { 
    if (!QueryPerformanceFrequency((LARGE_INTEGER*)&pcfreq))
	{ mexErrMsgTxt("Performance Counter not working");
	}
	tickspermsec= (double)pcfreq/1000.0;
    QueryPerformanceCounter((LARGE_INTEGER*)&pc);
    lastTime= ((double)pc)/tickspermsec;
    return;
  }

  QueryPerformanceCounter((LARGE_INTEGER*)&pc);
  elaps= ((double)pc)/tickspermsec - lastTime;
  toolate= 1;
  kbh= 0;
  while ((elaps<timeIncr) && !kbhit())
  { Sleep((DWORD)1);
    QueryPerformanceCounter((LARGE_INTEGER*)&pc); 
    elaps= ((double)pc)/tickspermsec - lastTime;
    toolate= 0;
  }
  if (toolate)                                          /* too late for sync */
  { out= elaps - timeIncr;
    lastTime= ((double)pc)/tickspermsec;
  }
  else
  { out= 0.0;
    lastTime+= timeIncr;
  }
  
  if (nlhs>0)
  { pArgOut= mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(pArgOut)= out;
  }
  if (nlhs>1)
  { if (kbhit())
    { pArgOut2= mxCreateDoubleMatrix(1, 1, mxREAL);
      *mxGetPr(pArgOut2)= getch();
    }
    else
    { pArgOut2= mxCreateDoubleMatrix(0, 0, mxREAL);
    }
  }
}
