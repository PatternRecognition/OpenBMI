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
  double timeIncr, elaps;
  bool out;
  long secIncr;
  int rc;

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

  gettimeofday(&thisTime, &tzone);
  elaps= (thisTime.tv_sec-lastTime.tv_sec)*1000.0 + 
    (thisTime.tv_usec-lastTime.tv_usec)/1000.0;
  
  out = (elaps>=timeIncr);

  if (out) {
    lastTime.tv_sec= thisTime.tv_sec;
    lastTime.tv_usec= thisTime.tv_usec;
  }

  if (nrhs>0)
  { pArgOut= mxCreateDoubleMatrix(1, 1, mxREAL);
    *mxGetPr(pArgOut)= out ? 1.0 : 0.0;
  }
}
