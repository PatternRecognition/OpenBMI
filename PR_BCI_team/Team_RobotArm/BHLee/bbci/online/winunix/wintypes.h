#ifndef WINTYPES_H
#define WINTYPES_H

typedef int HANDLE;
typedef long int DWORD;
typedef long int *LPDWORD;
typedef void *LPVOID;
typedef void *LPSECURITY_ATTRIBUTES;
typedef int BOOL;
typedef void VOID;
typedef const char *LPCTSTR;
typedef int SIZE_T;
typedef DWORD LPTHREAD_START_ROUTINE(LPVOID lpParameter);

#define WINAPI
#define INFINITE -1

#define FALSE 0
#define TRUE 1

#define WAIT_OBJECT_0	1
#define WAIT_TIMEOUT	2

#endif
