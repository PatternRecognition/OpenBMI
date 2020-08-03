/* Benjamin Blankertz */

#include <sys/time.h>
#include <time.h>
#include "mex.h"

#define pArgIn  prhs[0]
#define pArgOut plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  static struct timeval lastTime;
  static struct timezone tzone;
  struct timeval thisTime;
  double timeIncr, elaps, out;
  int rc, toolate;

  if (nrhs!=1)
    timeIncr= 0.0;
  else
    timeIncr= (double)*mxGetPr(pArgIn);

  if (timeIncr==0.0)
  { 
    tzone.tz_minuteswest= 0;
    tzone.tz_dsttime= 0;
    rc= gettimeofday(&lastTime, &tzone);
    if (rc)
    { mexErrMsgTxt("trouble with time function");
    }
    return;
  }

  rc= 0;
  elaps= 0.0;
  while (elaps<timeIncr)
  { gettimeofday(&thisTime, &tzone); 
    elaps= (thisTime.tv_sec-lastTime.tv_sec)*1000.0 + 
           (thisTime.tv_usec-lastTime.tv_usec)/1000.0;
    rc++;
  }
  if (rc==1)                                      /* too late for sync */
    out= elaps - timeIncr;
  else
    out= 0.0;

  lastTime.tv_usec= lastTime.tv_usec + (long)(timeIncr*1000.0);
  if (lastTime.tv_usec >= 1000000)
    { long carry;
      carry= (long)((double)lastTime.tv_usec/1000000.0);
      lastTime.tv_usec= lastTime.tv_usec - carry * 1000000.0;
      lastTime.tv_sec= lastTime.tv_sec + carry;
    }

  if (nlhs>0)
  { pArgOut= mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(pArgOut)= out;
  }
}
