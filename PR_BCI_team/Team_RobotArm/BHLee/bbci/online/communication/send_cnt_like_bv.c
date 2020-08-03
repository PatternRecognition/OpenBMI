/*
  send_cnt_like_bv.c

  This file defines a mex-Function to simulate a BrainVision server. 
  For data an calling conventions please see snd_cnt_like_bv.m or type
  'help send_cnt_like_bv' in Matlab
  
      INIT: 
         send_cnt_like_bv(cnt, 'init');
      DATA:
         pp= send_cnt_like_bv(cnt, pp);
         pp= send_cnt_like_bv(cnt, pp, mrkPos,mrkDesc);
 
      CLOSE:
         send_cnt_like_bv('close');

  Arguments:
   INIT:
      cnt         - the eeg data structure we use following fiels:
              cnt.fs - the sampling rate
            cnt.clab - the channel names
   DATA:
     cnt          - the eeg data structure we use following fiels:
              cnt.x  - the eeg data
     pp           - the position in the eeg data
     mrkPos       - the positions of the markers
     mrkDesc      - the description of the markers

 
  Compile:
      make_send_cnt_linke_bv
   
  ????/??/?? - ???
                - file created
  2008/04/22 - Max Sagebaum
                - file refactored and documented 
  
  (c) Fraunhofer FIRST.IDA 2008
*/
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include "myRDA.h"
#include "mex.h"

#ifdef _WIN32
/* WIN32 */
  #  include <winsock2.h>
  #  include <winbase.h>
  #  define CLOSESOCKET(s) closesocket(s);


#else	
/* UNIX */
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <sys/select.h>
  #include <sys/socket.h>
  #include <sys/wait.h>
  #include <sys/time.h>
  #include <unistd.h>
  #define CLOSESOCKET(s) close(s);

#endif


#define BV_PORT 51234l		/* the port bv sends data to */

/*
 * The static server information we need to send data to the client or
 * close the connection.
 */

static struct sockaddr_in addr;	/* my address information */
static int sockfd = -1;         /* the handle for the server*/
static int client = -1;         /* a handle for the connection  to the client*/

/*
 * The information we need to send the eeg data
 */
static int nChans;                    /* The channel count */
static ULONG nBlock;                  /* The x package we have send */
static double fs;                     /* The sampling frequency */

/*
 * FORWARD DECLARATIONS
 */ 

void snd_cnt_init(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]);

void snd_cnt_data(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]);

void snd_cnt_close(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]);

void snd_assert(bool aValue,const char* text);


void
mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
  /* check which mode we use */ 
  if (nrhs == 4 || (nrhs == 2 && (!mxIsChar(prhs[1])))) {
    snd_assert(nlhs == 1, "one output argument maximum");
    snd_assert(mxIsStruct(prhs[0]), "first argument must be struct");
    snd_assert(mxIsNumeric(prhs[1]), "second argument must be real scalar");
    snd_assert(mxGetM(prhs[1]) * mxGetN(prhs[1]) == 1,
       "second argument must be real scalar");
    
    snd_cnt_data(nlhs,plhs,nrhs,prhs);
  } else if (nrhs == 1 && mxIsChar(prhs[0])) {
    snd_assert(nlhs == 0, "no output argument expected");
    snd_cnt_close(nlhs,plhs,nrhs,prhs);
  } else if (nrhs == 2 && mxIsChar(prhs[1])) {
    snd_assert(mxIsStruct(prhs[0]), "first argument must be struct");
    snd_cnt_init(nlhs,plhs,nrhs,prhs);
  } else {
    mexErrMsgTxt("syntax error");
  }
}

/*************************************************************
 * This is the init function which creates a socket and acquires a
 * connection from one client.
 *************************************************************/
void snd_cnt_init(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {
  int nResult;                       /* see if a operation was successful*/ 
  const mxArray *IN_CNT;             /* the eeg data information*/
  mxArray *pArray;                   /* a temporary array for mex structure*/
  mxArray *pCntClab;                 /* The pointer to the channel names */
  
  int msgSize;                       /* The size of the start message */
  struct RDA_MessageStart *pMsgStart; /* The start message */
  double *pRes;                       /* pointer to pMsgStart->dResolutions */
  char *pLab;                         /* pointer ot pMsgStart->sChannelNames */
  int on = 1;
  
  int c;
  
	if (sockfd > 0) {
	    mexWarnMsgTxt ("Connection still open: closing the old one");
            snd_cnt_close(nlhs,plhs,nrhs,prhs);
	}
	sockfd = socket(PF_INET, SOCK_STREAM, 0);
	if (sockfd == -1) {
	    mexErrMsgTxt("socket failed");
	}
	
      /* on linux we need to set the reuse addres option */
      
      if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0)
        {
          perror("setsockopt(SO_REUSEADDR) failed");
        }

  /* setup the socket address structure */
	addr.sin_family = AF_INET;	/* host byte order */
	addr.sin_port = htons(BV_PORT);	/* short, network byte order */
	addr.sin_addr.s_addr = INADDR_ANY;	/* automatically fill with my IP */
	memset(&(addr.sin_zero),0, 8);	/* zero the rest of the struct */

	nResult =
	    bind(sockfd, (struct sockaddr *) &addr,
		 sizeof(struct sockaddr));
	if (nResult == -1) {
	    
	    printf("\nFailed to create a socket!\n\a");
            printf("%s", strerror(errno));
            mexErrMsgTxt("bind failed");
	} else
	    mexPrintf("binded to socket [%d]\n", sockfd);

	nResult = listen(sockfd, 1); /* we accept now one connection*/
	client = accept(sockfd, 0, 0); /* get the first connection, this method will block until we get one */
	mexPrintf("client: %d\n", client);

  /* send the start message */
  IN_CNT = prhs[0]; /* we assume that the struct has the right format,
                     * that is to say we do not check if for formats, types
                     * existence, etc.*/
  
  /* calculating the size of the message */
	pCntClab = mxGetField(IN_CNT, 0, "clab");
	nChans = mxGetN(pCntClab);
	msgSize = sizeof(struct RDA_MessageHeader) + sizeof(ULONG) +
	    sizeof(double) + nChans * sizeof(double); /* TODO: check if we have the correct size*/
	for (c = 0; c < nChans; c++) {
	    pArray = mxGetCell(pCntClab, c);
	    msgSize = msgSize + mxGetN(pArray) + 1; /* the size of the string plus \0 */
	}

  /* TODO: check if the pointers are right */
	pMsgStart = (struct RDA_MessageStart *) malloc(msgSize);
	pMsgStart->nSize = msgSize;
	pMsgStart->nType = 1; /* Magic number for RDA_MessageStart, see "myRDA.h" for mor information */
	pMsgStart->nChannels = nChans;
	pArray = mxGetField(IN_CNT, 0, "fs"); /* the sampling rate */
  fs = mxGetScalar(pArray);
	pMsgStart->dSamplingInterval = 1000000.0 / fs;
	pRes = pMsgStart->dResolutions;
	pLab = ((char *) pRes) + nChans * sizeof(double);
	for (c = 0; c < nChans; c++) {
	    pRes[c] = 0.1;
	    pArray = mxGetCell(pCntClab, c);
	    mxGetString(pArray, pLab, 255);
	    pLab += strlen(pLab) + 1;
	}

  /* send the message */
	nResult = sendto(client, (char*)pMsgStart, pMsgStart->nSize, 0,
			 (struct sockaddr *) &addr,
			 sizeof(struct sockaddr));
	if (nResult == -1) {
	    mexErrMsgTxt("sendto failed");
	}
  
  /* now we can start sending the data blocks */ 
	nBlock = 0;

	free(pMsgStart);
}

/*************************************************************
 * We will send one package of data
 *************************************************************/
void snd_cnt_data(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {
  int pp;                                   /* the position in the eeg data */
  int nPoints;                              /* how many data points we send */
  int nMarkers;                             /* how many markers we send */
  int nResult;                             /* see if an operation was successful */
  char* textBuff;                           /* pointer to the marker text */
  
  double *pPos;                             /* pointer to the marker positions */
  double *pSrc;                             /* pointer to the eeg data */
  int chanCountData;                        /* channel count of the eeg data */
  int msgSizeMarker;                        /* the siye of one marker */
  int msgSize;                              /* the size of the message */
  struct RDA_MessageData *pMsgData;                /* the message */
  short *pDst;                              /* pointer to the eeg data in the message */
  struct RDA_Marker *pMarker;               /* pointer to the markers */
  
  const mxArray *IN_CNT;
  const mxArray *IN_PP;
  const mxArray *IN_MRK_POS;
  const mxArray *IN_MRK_TOE;
  mxArray *pArray;                           /* temporary pointer */
  
  int c,t;
  
  IN_CNT = prhs[0];
  IN_PP = prhs[1];
  if (nrhs == 2) {
	    nMarkers = 0;
	} else {
      IN_MRK_POS = prhs[2];
      IN_MRK_TOE = prhs[3];
	    nMarkers = mxGetN(IN_MRK_POS);
	    pPos = mxGetPr(IN_MRK_POS);
	}
  
  pp = (int)mxGetScalar(IN_PP) - 1;
	
	nPoints = (ULONG) (0.040 * fs); /* send the data points for 40 ms */
  
  /* this is a static marker size we will cut off longer marker disciptions */
	msgSizeMarker = 3 * sizeof(ULONG) + sizeof(long) + (9 + 5) * sizeof(char);
	msgSize = sizeof(struct RDA_MessageHeader) + sizeof(ULONG) +
	    sizeof(double) + nChans * nPoints * sizeof(short) +
	    nMarkers * msgSizeMarker; /* TODO: check size of the message */
  
	pMsgData = (struct RDA_MessageData *) malloc(msgSize);
	pMsgData->nSize = msgSize;
	pMsgData->nType = 2; /* Magic number for RDA_MessageData, see "myRDA.h" for mor information */
	pMsgData->nBlock = ++nBlock;
	pMsgData->nPoints = nPoints;
	pMsgData->nMarkers = nMarkers;
  
	pDst = pMsgData->nData;   /* set the pointer to the data */
	pMarker = (struct RDA_Marker *) (pDst + nChans * nPoints); /* calculate the pointer to the markers */
	pArray = mxGetField(IN_CNT, 0, "x");
	chanCountData = mxGetM(pArray);
	pSrc = mxGetPr(pArray);
  
  /* set the eeg data */
	for (t = pp; t < pp + nPoints; t++) {
    for (c = 0; c < nChans; c++) {
      *pDst = (short) (10.0 * pSrc[c * chanCountData + t]);
      pDst++;
    }
	}
  /* set the markers */
	for (t = 0; t < nMarkers; t++) {
      pArray = mxGetCell(IN_MRK_TOE,t);
	    pMarker->nSize = msgSizeMarker;
	    pMarker->nPosition = pPos[t] - pp;
	    pMarker->nPoints = 1;
	    pMarker->nChannel = -1;
	    textBuff = pMarker->sTypeDesc;

      mxGetString(pArray,textBuff + 9,5); /* write the description of the 
                                          * marker to the last 4 chars
                                          * plus \0 */
	    if (textBuff[9] == 'R') {
         sprintf(textBuff, "Response\0");
	    } else if(textBuff[9] == 'S') {
         sprintf(textBuff, "Stimulus\0");
      } else {
          /*write empty marker */
          sprintf(textBuff, "        \0");
          sprintf(textBuff + 9, "    \0");
        }
      
      /* get the pointer to the next marker */
	    pMarker = (struct RDA_Marker *) (((char *) pMarker) + msgSizeMarker);
	}
	/*
	 * mexPrintf("calc: %d, actual: %d\n", msgSize, (int)((char*)pMarker - 
	 * (char*)pMsgData));
	 */

	nResult = sendto(client, pMsgData, pMsgData->nSize, 0,
			 (struct sockaddr *) &addr,
			 sizeof(struct sockaddr));
	if (nResult == -1) {
	    mexErrMsgTxt("sendto failed");
	}

  /* write back the data position */
	plhs[0] = mxCreateDoubleScalar(pp + nPoints + 1);
	free(pMsgData);
}

/*************************************************************
 * We close the connection to the client and we will close the
 * server.
 *************************************************************/
void snd_cnt_close(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]) {
  /* if (nrhs == 0) {	 brute force closing  do we need this ???
	    for (c = 50; c > 10; c--)
		close(c);
	    sockfd = client = -1;
	} */
  
  int nResult;                             /* see if an operation was successful */
  
  if (client > 0) {
    nResult = CLOSESOCKET(client);
    if (nResult == 0) {
      mexPrintf("client [%d] closed\n", client);
      client = -1;
    } else {
      mexErrMsgTxt("cannot close client");
    }
	}
  
	if (sockfd > 0) {
    nResult = CLOSESOCKET(sockfd);
    if (nResult == 0) {
      mexPrintf("server [%d] closed\n", sockfd);
      sockfd = -1;
	  } else {
      mexErrMsgTxt("cannot close server");
	  }
	}
}

/************************************************************
 *
 * checks for errors
 *
 ************************************************************/
void snd_assert(bool aValue,const char *text) {
  if(!aValue) {
    mexErrMsgTxt(text);
  }
}
