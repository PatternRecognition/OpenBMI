#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include "winevents.h"

char *strdup(const char *s)
{
  if(s) {
    int len = strlen(s) + 1;
    char *a = malloc(len * sizeof(char));
    strcpy(a, s);
    return a;
  }
  else
    return 0;
}

/* local structs */
struct __event_t {
  pthread_mutex_t mut;
  pthread_cond_t cond;
  int signalled;
  char *name;
};

typedef struct __event_t event_t;


extern HANDLE CreateEvent(
  LPSECURITY_ATTRIBUTES lpEventAttributes,
  BOOL bManualReset,
  BOOL bInitialState,
  LPCTSTR lpName
)
{
  event_t *e = (event_t*) malloc(sizeof(event_t));
  pthread_mutex_init(&e->mut, NULL);
  pthread_cond_init(&e->cond, NULL);
  e->signalled = 0;
  e->name = strdup(lpName);
  return (HANDLE)e;
}


extern DWORD WaitForSingleObject(
  HANDLE hHandle,
  DWORD dwMilliseconds
)
{
  event_t *e = (event_t*)hHandle;

#ifdef DEBUG
  printf("Waiting for event %s ", e->name);
#endif

  pthread_mutex_lock(&e->mut);

#ifdef DEBUG
  printf(" at level %d\n", e->signalled);
#endif

  if(e->signalled) {
    e->signalled--;
    pthread_mutex_unlock(&e->mut);
    
    return WAIT_OBJECT_0;
  }

  if (dwMilliseconds == INFINITE) {
    pthread_cond_wait(&e->cond, &e->mut);
  }
  else {
    struct timespec timeout;
    struct timeval now;
    int isec = dwMilliseconds / 1000;
    int insec = (dwMilliseconds % 1000) * 1000000;
    
    gettimeofday(&now, NULL);
    
    timeout.tv_sec = now.tv_sec;
    timeout.tv_nsec = now.tv_usec;
    
    timeout.tv_nsec += insec;
    if(timeout.tv_nsec > 1000000000) {
      timeout.tv_nsec -= 1000000000;
      timeout.tv_sec++;
    }
    timeout.tv_sec += isec;
    
    if(pthread_cond_timedwait(&e->cond, &e->mut, &timeout)
       == ETIMEDOUT) {
      pthread_mutex_unlock(&e->mut);
      return WAIT_TIMEOUT;
    }
  }
  e->signalled--;
  pthread_mutex_unlock(&e->mut);

  return WAIT_OBJECT_0;
}


extern BOOL SetEvent(
  HANDLE hEvent
)
{
  event_t *e = (event_t*)hEvent;
  pthread_mutex_lock(&e->mut);
  pthread_cond_signal(&e->cond);
  e->signalled++;
  pthread_mutex_unlock(&e->mut);

#ifdef DEBUG
  printf("Signalled event %s to level %d\n", e->name, e->signalled);
#endif
}


/* in our case only defined for events!
// if this is going to be extended, use unions and a type field, PLEEZE!
*/
extern BOOL CloseHandle(
  HANDLE hObject
)
{
  event_t *e = (event_t*)hObject;

  pthread_mutex_destroy(&e->mut);
  pthread_cond_destroy(&e->cond);

  free(e);
}
