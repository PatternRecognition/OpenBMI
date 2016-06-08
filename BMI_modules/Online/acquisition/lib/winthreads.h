#ifndef WINTHREADS_H
#define WINTHREADS_H

#include "wintypes.h"

extern HANDLE CreateThread(
  LPSECURITY_ATTRIBUTES lpThreadAttributes,
  SIZE_T dwStackSize,
  LPTHREAD_START_ROUTINE lpStartAddress,
  LPVOID lpParameter,
  DWORD dwCreationFlags,
  LPDWORD lpThreadId
);

extern BOOL TerminateThread(
  HANDLE hThread,
  DWORD dwExitCode
);

VOID ExitThread(
  DWORD dwExitCode
);

extern VOID Sleep(
  DWORD dwMilliseconds
);

#endif

