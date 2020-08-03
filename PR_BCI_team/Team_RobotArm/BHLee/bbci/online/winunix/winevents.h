#ifndef WINEVENTS_H
#define WINEVENTS_H

#include "wintypes.h"

extern HANDLE CreateEvent(
  LPSECURITY_ATTRIBUTES lpEventAttributes,
  BOOL bManualReset,
  BOOL bInitialState,
  LPCTSTR lpName
);

extern DWORD WaitForSingleObject(
  HANDLE hHandle,
  DWORD dwMilliseconds
);

extern BOOL SetEvent(
  HANDLE hEvent
);

extern BOOL CloseHandle(
  HANDLE hObject
);

#endif
