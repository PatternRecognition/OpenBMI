#include <stdio.h>
#include <unistd.h>
#include "winthreads.h"
#include "winevents.h"

DWORD threadproc(LPVOID params)
{
  int i;

  for(i = 0; i < 10; i++) {
    printf("Happy thread\n");
    Sleep(500);
  }
  ExitThread(0);
}

HANDLE waitEvent;
HANDLE okayEvent;

DWORD waitingproc(LPVOID params)
{
  printf("Waiting...\n");
  WaitForSingleObject(waitEvent, INFINITE);

  Sleep(6000);

  printf("Sending ok\n");
  SetEvent(okayEvent);
  printf("Waiting...\n");
  WaitForSingleObject(waitEvent, INFINITE);
  
  Sleep(1000);

  printf("Sending ok\n");
  SetEvent(okayEvent);
  printf("done\n");
}

int main(int argc, char **argv)
{
  int ms = 1000;
  int i;

  // Test Events
  waitEvent = CreateEvent(NULL, FALSE, FALSE, "no name");
  okayEvent = CreateEvent(NULL, FALSE, FALSE, "no name");

  printf("Creating thread\n");
  CreateThread(NULL, 0, waitingproc, NULL, 0, NULL);

  Sleep(1000);
  printf("Singalling\n");
  SetEvent(waitEvent);
  printf("Waiting for ok\n");
  if(WaitForSingleObject(okayEvent, 500) == WAIT_TIMEOUT) {
    printf("Timed out, waiting again!\n");
  }
  WaitForSingleObject(okayEvent, INFINITE);

  Sleep(5000);

  printf("And signalling again\n");
  SetEvent(waitEvent);
  Sleep(5000);
  printf("Waiting for ok\n");
  WaitForSingleObject(okayEvent, INFINITE);

  CloseHandle(waitEvent);
  CloseHandle(okayEvent);

  Sleep(2000);

  // Testing threads and Sleep
  if(argc == 2) ms = atoi(argv[1]);

  CreateThread(NULL, 0, threadproc, NULL, 0, NULL);

  for(i = 0; i < 3; i++) {
    printf("Waiting...\n");
    Sleep(1000);
  }

  Sleep(2500);

  return;
}
