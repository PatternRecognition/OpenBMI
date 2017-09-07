#include "mex.h"
#include <conio.h>
#include "ioPort.c"

#define pAddr prhs[0]
#define pData prhs[1]
#define pSleepTime prhs[2]
#define pDelayTime prhs[3]


/* ppWrite(IO_ADDR, Value) or ppWrite(IO_ADDR, Value, SleepTime) is used 
   to send a trigger value to the parallel port that has 
   approx. 10 milliseconds duration, maybe a bit longer. Before and after this interval,
   the parallel port has to be set back to the value of 0.
   ppWrite is usually called by a feedback routine during an BCI experiment.
   Parallel port values are to be read by e.g. BrainVision Recorder.

   As other triggers can be send during the 10ms interval, ppWrite has to check the 
   current value on the parallel port at the end of the 10ms before sending a value of 0.
   This can unfortunately NOT be done by query to the parallel port directly (as
   the parallel port ist full duplex), but only by storing the value in a static variable
   called "portvalue".
 
   ppWrite uses now the file ioPort.c for the io functionality. In ioPort 
   the parallel port is accesed in windows nt save way.
 
   If you have problems with the parallelport see ioPort.c.
     
   ToDo: Prevent funny things happening by securing a mutex lock to the access of the 
   static "portvalue".
   
   Benjamin, Michael and Mikio 27.10.2006
   edit Max Sagebaum           02.11.2007       Added nt save io functions
   edit Max Sagebaum           16.05.2008       Enabled user to set the sleep time 
   edit Max Sagebaum           16.11.2011       Enabled user to set a delay
*/

typedef unsigned short ioaddr_t;
typedef unsigned char portvalue_t;

struct threadparams {
  ioaddr_t portaddr;
  portvalue_t sendvalue;
  int sleepTime;
  int delayTime;
  HINSTANCE hLib;  
};

static portvalue_t currentvalue = 0;

/* Thread that sets signal on parallel port to zero after 10 ms of sleeping */
DWORD WINAPI ThreadFunc( LPVOID lpParam )
{
  int sleepTime;
  int delayTime;
  
  struct threadparams *p = lpParam;
  ioaddr_t portaddr = p->portaddr;
  portvalue_t sendvalue = p->sendvalue;
  sleepTime = p->sleepTime;
  delayTime = p->delayTime;
  
  if(0 != delayTime) {
      /* send a value to the parallel prot after the delay time. */
      Sleep(delayTime);
      
      Out32(portaddr,sendvalue);
      currentvalue = sendvalue;
  }
  
  /* after sleeping 10msec: check, if the parallel port still shows the sent value
     by checking the static "currentvalue". 
     If it does, set parallel port back to zero.
     If it does not, hands off! ( it probably means that some other process has sent 
     a trigger to the parallel port in the meantime) 
  */
  Sleep(sleepTime);   
  
  /* ToDo: put mutex lock around the following three lines */
  if (currentvalue == sendvalue) {
    Out32(portaddr, 0);
    currentvalue = 0;
  } 
  else {
    //mexPrintf("ppWrite: trigger overlap on parport detected.\n");
  }
  
  deleteLibrary(p->hLib);
  free(p);
  return 0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  struct threadparams *p;
  
  if (nrhs < 2 && nrhs > 4)
	mexErrMsgTxt("Two or three or four input arguments required");
  if (nlhs>0)
	mexErrMsgTxt("No output argument allowed");

  p = malloc(sizeof(*p));
  p->hLib = createLibrary();
	
  /* get parameters from mex format and store into struct p */
  p->portaddr= (unsigned short)*mxGetPr(pAddr);
  p->sendvalue= (unsigned char)*mxGetPr(pData);
  if(nrhs >= 3) {
    p->sleepTime = (int)mxGetScalar(pSleepTime);
  } else {
    p->sleepTime = 10;
  }
  
  if(nrhs >= 4) {
    p->delayTime = (int)mxGetScalar(pDelayTime);
  } else {
    p->delayTime = 0;
  }
  
  /* mexPrintf("Intent to write %d, current value is %d\n", p->sendvalue, currentvalue); */
  /* ToDo: Put mutex lock around the following two lines */
  
  /* Only set the value if the delay is zero, otherwise the thread will set the value */
  if( 0 == p->delayTime) {
    Out32(p->portaddr, p->sendvalue);
    currentvalue = p->sendvalue;
  }
    
  /* mexPrintf("Wrote %d\n", p->sendvalue); */
  /* Start thread that resets parport value to zero in 10ms */
  CreateThread( NULL, 0, &ThreadFunc, (LPVOID) p, 0, 0);
}
