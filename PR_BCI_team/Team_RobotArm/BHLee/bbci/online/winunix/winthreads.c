#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include "winthreads.h"

typedef void *pthread_fct(void*);

HANDLE CreateThread(
  LPSECURITY_ATTRIBUTES lpThreadAttributes,
  SIZE_T dwStackSize,
  LPTHREAD_START_ROUTINE lpStartAddress,
  LPVOID lpParameter,
  DWORD dwCreationFlags,
  LPDWORD lpThreadId
)
{
  pthread_t tid;

  if(pthread_create(&tid, NULL, (pthread_fct*)lpStartAddress, lpParameter)) {
    perror("winthreads: pthread_create faild");
    return (HANDLE)0;
  }
  else
    return (HANDLE)tid;
}

BOOL TerminateThread(
  HANDLE hThread,
  DWORD dwExitCode
)
{
  pthread_t tid = (pthread_t)hThread;

  pthread_cancel(tid);
}

VOID ExitThread(
  DWORD dwExitCode
)
{
  pthread_exit((void*)dwExitCode);
}

VOID Sleep(
  DWORD dwMilliseconds
)
{
  struct timespec ts;
  struct timespec rem;

  ts.tv_sec = dwMilliseconds / 1000;
  ts.tv_nsec = (dwMilliseconds % 1000) * 1000000;
  
  while(!nanosleep(&ts, &rem) && errno == EINTR 
	&& rem.tv_sec > 0 && rem.tv_nsec > 0) {
    printf("Sleeping for another %d.%03d s\n",
	   ts.tv_sec, ts.tv_nsec/1000000);
    ts = rem;
    nanosleep(&ts, &rem);
  }
}



