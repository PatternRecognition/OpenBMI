/*
  brainserver.h

  this file defines functions for communication with the brainvision
  server.

  This has been written by Mikio Braun, mikio@first.fhg.de
  (c) Fraunhofer FIRST.IDA 2005
*/

/*
  This file actually consists of two parts:
  
  (1) establishing a connection and reading packets
  (2) a thread which collects the data in the background

  It just didn't make sense to split the file, because the second part
  is pretty useless without the first part.

  Note though, that some global variables which are used for the
  thread are defined at the beginning of the second part. Just search
  for "Threading"

  -Mikio
 
  - 2008/09/18 - Max Sagebaum
                 - I checked the warning messages and tried to fix them. In
                   order to remove the mexPrintf and printf messages I had to
                   add mex.h to the header file. Therefore the typedef of bool
                   is not needed any longer. The working mexPrintf and prinf
                   functions produced heavy output in the threading part of
                   the file. I documented them out.
*/

#ifdef _WIN32
/* WIN32 */
#  include <winsock2.h>
#  include <winbase.h>
#  include <stdio.h>

#  define CLOSESOCKET(s) do { closesocket(s); brainserver_socket = -1; } while(0)
#else	
/* UNIX */
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <unistd.h>
#include "../../../online/winunix/winthreads.h" /* These are some portability layers */
#include "../../../online/winunix/winevents.h"  /* for WinThreads and Events */
#define CLOSESOCKET(s) do { close(s); brainserver_socket = -1; } while(0)

#endif

#include "brainserver.h"
#include "headerfifo.h"

/*
  implementation specific data types
*/

/* A union of the different message types.
   To check for the type do header.nType; */
union RDA_Message {
  struct RDA_MessageHeader header;
  struct RDA_MessageStart start; /* header.nType = 1*/ 
  struct RDA_MessageData data;   /* header.nType = 2 (int16) or 4 (int32) */
  struct RDA_MessageStop stop;   /* header.nType = 3 */
};

/*
  forward references of local functions
*/
void dumpRDAMarker(struct RDA_Marker *pma);

/* threading forward references */
void printThreadState();
DWORD WINAPI pollThread(LPVOID lpParameter);
int startPollThread();
int stopPollThread();
int beginFIFO();
int finishedFIFO();


/*
  the main socket
*/

int brainserver_socket = 0;

/*----------------------------------------------------------------------

Main Functions

----------------------------------------------------------------------*/


/*
  initializing the socket
 */

int initConnection(const char *bv_hostname,
		   struct RDA_MessageStart **pMsgStart)
{
  struct sockaddr_in addr;                 /* my address information */
  struct hostent *he;
  int failed, waiting, nResult;

  *pMsgStart = NULL;

#ifdef _WIN32
  /* Setup sockets for windows */
  {
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD( 2, 2 );
    
    /* Windows Socketschnittstelle vorbereiten */
    err = WSAStartup( wVersionRequested, &wsaData );
    if ( err != 0 ) return -1;
    if ( LOBYTE( wsaData.wVersion ) != 2 
	 || HIBYTE( wsaData.wVersion ) != 2 ) 	{
      WSACleanup( );
      return -1; 
    }
  }
#endif

  /* get the host info */
  if ((he = gethostbyname(bv_hostname)) == NULL) {
      mexWarnMsgTxt("acquire_bv: gethostbyname failed.");
      CLOSESOCKET(brainserver_socket);
      return -2;
  }
      
  /* open a socket */
  if ((brainserver_socket = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
      mexWarnMsgTxt("acquire_bv: Couldn't open socket.");
      CLOSESOCKET(brainserver_socket);
      return -3;
  }
  
  addr.sin_family = AF_INET;         /* host byte order */
  addr.sin_port = htons(BV_PORT);    /* short, network byte order */
  addr.sin_addr = *((struct in_addr *)he->h_addr);

  /* connect */
  if (connect(brainserver_socket,
	      (struct sockaddr *)&addr, 
	      sizeof(struct sockaddr)) 
      == -1) {
    mexWarnMsgTxt("acquire_bv: cannot connect to server");
    CLOSESOCKET(brainserver_socket);
    return -1;
  }
  else
    mexPrintf("connected to %s: %s -> socket [%d]\n", bv_hostname, 
	      inet_ntoa(addr.sin_addr), brainserver_socket);
  
  /* Keep reading until a whole header was received. */
  failed = 0;
  waiting = 1;
  while (waiting) {
    struct RDA_MessageHeader *pHeader = 0;

    nResult = getServerMessage(brainserver_socket, &pHeader);
    if (nResult > 0) {
      if(pHeader->nType == 1) { /* Header */
	*pMsgStart = (struct RDA_MessageStart*)pHeader;
        pHeader = 0;
	waiting = 0;
      }
      else if(pHeader->nType == 3) { /* Stop signal */
	mexWarnMsgTxt("transmission was stopped\n");
	failed = 1;
	waiting = 0;
      }
    }
    else if (nResult <= 0) {
      mexWarnMsgTxt("error in transmission.\n");
      failed = 1;
      waiting = 0;
    }
    if (pHeader) free(pHeader);
  }
  
  if (failed) { 
    CLOSESOCKET(brainserver_socket);
    return -1;
  }

#ifdef AC_THREADED
  return startPollThread();
#else
  return 0;
#endif
}

int DetermineElementSize(int nType)
{    
    switch (nType) {
        case 1:
            mexWarnMsgTxt("A start packet shouldn't appear here");
            return 0;
            break;
        case 2:
            return 2;
            break;
        case 3:
            mexWarnMsgTxt("Stop packet received");
            return 0;
            break;
        case 4:
            return 4;
            break;
        default:
            mexErrMsgTxt("Unknown packet type");
    }
}

/*
  reading data
*/

int getData(struct RDA_MessageData **pMsgData, int *blocksize,
        int lastBlock, ULONG nChannels, int *pElementSize)
 {
    int nResult;
    
#ifndef AC_THREADED
    int failed, waiting;

    /* read messages till we get a data message */
    failed = 0;
    waiting = 1;

    while (waiting) {
        struct RDA_MessageHeader *pHeader = 0;

        nResult = getServerMessage(brainserver_socket, &pHeader);
        if (nResult > 0) {
            if (pHeader->nType == 2 || pHeader->nType == 4) {
                *pMsgData = (struct RDA_MessageData*)pHeader;
                if ((*pMsgData)->nBlock != lastBlock) {
                    pHeader = 0;
                    waiting = 0;
                }
                *pElementSize = DetermineElementSize(pHeader->nType);
            }
            else if(pHeader->nType == 3) { /* Stop Signal */
                mexWarnMsgTxt("transmission was stopped\n");
                failed = 1;
                waiting = 0;
            }
        }
        else if (nResult <= 0) {
            mexWarnMsgTxt("error in transmission.\n");
            failed = 1;
            waiting = 0;
        }
        if (pHeader) free(pHeader);
    }

    if (failed) {
        CLOSESOCKET(brainserver_socket);
        return -1;
    }
    else
        return 0;
#else /* read out data from the thread */

    nResult = beginFIFO();
    if(nResult == -1) {
       printf("Thread not running\n");
       printThreadState();
       return -1;
    }
    else {
      /* Traverser the FIFO two times:
       * 1) count Points and Markers
       * 2) concat everything
       */
        struct RDA_MessageHeader *pHeader;
        struct headerFIFONode *n;
        struct RDA_MessageData *pmd;

        /* count everything */
        int nPoints = 0;
        int nMarkers = 0;
        int nBlock = 0;

        int m;

        int structSize = sizeof(struct RDA_MessageData);
	/*  - sizeof(short) - sizeof(struct RDA_Marker); */
        int markerOffset = 0;

        if (fifo.last)
            *pElementSize = DetermineElementSize(fifo.last->payload->nType);
        else
            *pElementSize = 2;

        for(n = fifo.last; n; n = n->prev) {
            pHeader = n->payload;
            if(pHeader->nType == 2 || pHeader->nType == 4) {
                pmd = (struct RDA_MessageData*)pHeader;
                nPoints += pmd->nPoints;
                *blocksize = pmd->nPoints;
                nMarkers += pmd->nMarkers;        
                nBlock = pmd->nBlock;

                structSize += *pElementSize * nChannels * pmd->nPoints;
                for(m = 0; m < pmd->nMarkers; m++) {
                    struct RDA_Marker *pma = getMarker(pmd, nChannels, m, *pElementSize);
                    structSize += pma->nSize;
                }
            }
        }

        /* allocate a message structure of appropriate size */
        *pMsgData = malloc(structSize);
        if(!*pMsgData) {
            printf("Out of Memory!\n");
            finishedFIFO();
            return -1;
        }
        
        (*pMsgData)->nBlock = nBlock;
        (*pMsgData)->nPoints = nPoints;
        (*pMsgData)->nMarkers = nMarkers;

        /* Now collect the data */
        {
            char *pdst = (char*)(*pMsgData)->nData;
            struct RDA_Marker *mdst = 
                (struct RDA_Marker *)(pdst + *pElementSize * nChannels * nPoints);

            while(pHeader = headerFIFOpop(&fifo)) {
                if(pHeader->nType == 2 || pHeader->nType == 4) {
                    int psize;
                    pmd = (struct RDA_MessageData*)pHeader;

                    psize = *pElementSize * nChannels * pmd->nPoints;
                    memcpy(pdst, pmd->nData, psize);
                    pdst += psize;

                    for (m = 0; m < pmd->nMarkers; m++) {
                        struct RDA_Marker *pm = getMarker(pmd, nChannels, m, *pElementSize);
                        pm->nPosition += markerOffset;
                        memcpy(mdst, pm, pm->nSize);
                        mdst = (struct RDA_Marker*)((char*)mdst + pm->nSize);
                    }
                    markerOffset += pmd->nPoints;
                }
                free(pHeader);
            }
        }
    }
    finishedFIFO();

    return 0;
#endif
}


/*
  closing the connection
*/

void closeConnection()
{
#ifdef AC_THREADED
    stopPollThread();
#endif
    
    if (brainserver_socket>0) {
      CLOSESOCKET(brainserver_socket);
    } else {
      mexPrintf("socket [%d] closed already\n", brainserver_socket);
    }
}


/*----------------------------------------------------------------------
  
Auxillary functions

----------------------------------------------------------------------*/

/* Get message from server, if available                               
   returns 0 if no data, -1 if error, -2 if server closed,  > 0 if ok.
   e this fcn is slightly adapted from Henning Nordholz (BrainVision)    */
int getServerMessage(int sock, struct RDA_MessageHeader** ppHeader)
{
  /* this function basically reads an RDA_MessageHeader, 
     and then the rest specified 
   */
  struct timeval tv; 
  fd_set readfds;
  struct RDA_MessageHeader header;
  char* pData;
  int nResult, nReq;
  bool bFirstRecv;
  
  /* wait for something to happen on the socket or timeout */
  tv.tv_sec = 5; tv.tv_usec = 0;    /* 5 s. */
  FD_ZERO(&readfds);
  FD_SET(sock, &readfds);
  nResult = select(sock+1, &readfds, NULL, NULL, &tv);
  if (nResult != 1) return nResult;

  pData = (char*)&header;
  bFirstRecv = true;
  nReq = sizeof(header);
  
  /* Read out a whole RDA_MessageHeader structure */
  while (nReq > 0) 
  {
    nResult = recv(sock, pData, nReq, 0);

    /* When select() succeeds and recv() returns 0 
       the server has closed the connection.        */
    if (nResult == 0 && bFirstRecv)  return -2;

    bFirstRecv = false;

    /* still some bytes left? */
    if (nResult < 0) return nResult;
    nReq -= nResult;
    pData += nResult;
  }

  /* copy the header to ppHeader */
  *ppHeader = (struct RDA_MessageHeader *)malloc(header.nSize);
  if (!*ppHeader) return -1;

  memcpy(*ppHeader, &header, sizeof(header));

  /* read the actual data */
  pData = (char*)*ppHeader + sizeof(header);

  nReq = header.nSize - sizeof(header);
  while (nReq > 0)
  {
    nResult = recv(sock, pData, nReq, 0);
    if (nResult < 0) {
      free(*ppHeader);
      *ppHeader = 0;
      return nResult;
    }
    nReq -= nResult;
    pData += nResult;
  }

  return 1;
}


struct RDA_Marker *getMarker(struct RDA_MessageData *pmd, 
			     int nChannels, int idx, int ElementSize)
{
    if (0 <= idx && idx < pmd->nMarkers) {
        struct RDA_Marker *pm 
	  = (struct RDA_Marker *)((char*)pmd->nData + ElementSize * pmd->nPoints * nChannels);
        for(;idx; idx--) {
            pm = (struct RDA_Marker *)((char*)pm + pm->nSize);
        }
        return pm;
    }
    else
        return 0;
}


void dumpRDAMarker(struct RDA_Marker *pma)
{
    char *c;
    printf("nSize:     %d\n", pma->nSize);
    printf("nPosition: %d\n", pma->nPosition);
    printf("nChannel:  %d\n", pma->nChannel);
    printf("sTypeDesc: ");
    for(c = pma->sTypeDesc; c - (char*)pma < pma->nSize; c++) {
        if(isprint(*c)) printf("%c", *c);
        else printf(".");
    }
    printf("\n");
}

/**********************************************************************

Threading

**********************************************************************/

#ifdef AC_THREADED

/*
  Global variables
*/
enum {
    TS_INIT,
    TS_RUNNING,
    TS_WAITING,
    TS_ERROR,
    TS_STOPPED
}  threadStatus = TS_INIT;


enum {
    TR_CLEAR,
    TR_WAIT,
    TR_QUIT
} threadRequest = TR_CLEAR;


HANDLE pollThreadWait, pollRequestWait;


HANDLE threadHandle;



/*
  general thread control
*/

int startPollThread()
{
  /* Create some waiting threads */
    pollThreadWait = CreateEvent(NULL, FALSE, FALSE, "pollThreadWait");
    pollRequestWait = CreateEvent(NULL, FALSE, FALSE, "pollRequestWait");

    threadRequest = TR_CLEAR;
    threadHandle = CreateThread(NULL, 0, &pollThread, NULL, 0, NULL);

    WaitForSingleObject(pollRequestWait, INFINITE);

    if (threadStatus != TS_RUNNING)
        return -1;
    else    
        return 0;
}
 

int stopPollThread()
{
    int numele = 0;
    int count;

    struct RDA_MessageHeader *header;

    for(count = 2; (count > 0) && (threadStatus != TS_STOPPED); count --) {
        threadRequest = TR_QUIT;
        WaitForSingleObject(pollRequestWait, 500);
    }

    /* printf("Terminating thread!\n");*/

    if (!count) TerminateThread(threadHandle, 0);

    CloseHandle(pollThreadWait);
    CloseHandle(pollRequestWait);

    while(header = headerFIFOpop(&fifo)) { free(header); numele++; }; 

    if (threadStatus != TS_STOPPED)
        return -1;
    else    
        return 0;
}


void printThreadState()
{
    printf("threadStatus: ");
    switch(threadStatus) {
    case TS_INIT: printf("TS_INIT"); break;
    case TS_RUNNING: printf("TS_RUNNING"); break;
    case TS_WAITING: printf("TS_WAITING"); break;
    case TS_ERROR: printf("TS_ERROR"); break;
    case TS_STOPPED: printf("TS_STOPPED"); break;
    default: printf("UNKNOWN!!!");
    };

    printf(" threadRequest: ");
    switch(threadRequest) {
    case TR_CLEAR: printf("TR_CLEAR"); break;
    case TR_WAIT: printf("TR_WAIT"); break;
    case TR_QUIT: printf("TR_QUIT"); break;
    default: printf("UNKNOWN!!!");
    }
    printf("\n");
}


/*
  FIFO access synchronization
  
  Call these two functions when you want to access the fifo. The
  thread will be blocked until endFIFO() is called.
*/

int beginFIFO()
{
  if (threadStatus != TS_RUNNING) {
    return -1;
  }
  else {
    threadRequest = TR_WAIT;
    do {
      WaitForSingleObject(pollRequestWait, 1000);
      if (threadStatus == TS_ERROR)
	return -1;
    } while(threadStatus != TS_WAITING);
    
    return 0;
  }
}


int finishedFIFO()
{
  if (threadStatus != TS_WAITING) {
    return -1;
  }
  else {
    SetEvent(pollThreadWait);
    WaitForSingleObject(pollRequestWait, INFINITE);
    return 0;
  }
}


/*
  This is the actual thread.

  It polls the server for data and stores it in a doubly linked list.
*/

DWORD WINAPI pollThread(LPVOID lpParameter)
{
  /* keep polling the server and store everything in a FIFO. */
  /* after each message wait if somebody wants to read out the whole list. */
  /* in that case wait. Then proceed. */
  int lastBlock = -1;
  int result;
  
  struct RDA_MessageHeader *header = 0;
  
  /*printf("Thread running!\n");*/
  
  threadStatus = TS_RUNNING;
  SetEvent(pollRequestWait);

  while(1) {
    /* read a message and push it into the fifo */
    /* only data packets are pushed */
    header = 0;
    
    result = getServerMessage(brainserver_socket, &header);
    
    if (result > 0) {
      if (header->nType == 2 || header->nType == 4) {
	int block = ((struct RDA_MessageData *)header)->nBlock;

	if (block != lastBlock) {
	  headerFIFOpush(&fifo, header);
	  header = 0;
	  block = lastBlock;
	}
      }
      else if(header->nType == 3) { /* stopped */
	/*printf("Thread: Read STOP. Stopping\n");*/
	threadStatus = TS_ERROR;
	return -1;
      }
      if (header) free(header);
    }
    else {
      printf("Thread: getServerMessage return <= 0. Stopping\n");
      threadStatus = TS_ERROR;
      return -1;
    }
    
    /* If a wait has been requested, signal the waiting thread to poll */
    /* the fifo and wait for the signal that polling is over. */
    if (threadRequest == TR_WAIT) {
      threadStatus = TS_WAITING;
      SetEvent(pollRequestWait);
      WaitForSingleObject(pollThreadWait, INFINITE);
      threadStatus = TS_RUNNING;
      threadRequest = TR_CLEAR;
      SetEvent(pollRequestWait);
    }
    if (threadRequest == TR_QUIT) {
      /*printf("Thread is stopped...\n");*/
      threadStatus = TS_STOPPED;
      SetEvent(pollRequestWait);
      return 0;
    }
  }
}

#endif /* AC_THREADED */
/* end of brainserver.c */
