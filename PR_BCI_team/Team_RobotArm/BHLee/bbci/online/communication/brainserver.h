/*
  brainserver.h

  this header file defines functions for communication with the
  brainvision server.

  This has been written by Mikio Braun, mikio@first.fhg.de
  (c) Fraunhofer FIRST.IDA 2005
*/

#ifndef BRAINSERVER_H
#define BRAINSERVER_H

#include "myRDA.h"
#include "mex.h"

/*
  The brainvision server port number
 */
#ifndef BV_PORT
#define BV_PORT 51234
#endif

/*
  Error codes
*/

#define IC_OKAY                     0
#define IC_ERROR                    -1
#define IC_GETHOSTBYNAME_FAILED     -2
#define IC_OPENSOCKET_FAILED        -3

/*
  external variables
*/
extern int brainserver_socket;

/*
  The main access functions
*/
extern int
initConnection(const char *bv_hostname, struct RDA_MessageStart **pMsgStart);

extern int
getData(struct RDA_MessageData **pMsgData, int *blocksize, 
	int lastBlock, ULONG nChannels, int *pElementSize);

extern void
closeConnection();

/*
  Functions for more specific messages.
*/

extern int 
getServerMessage(int sock, struct RDA_MessageHeader** ppHeader);

extern struct RDA_Marker *
getMarker(struct RDA_MessageData *pmd, int nChannels, int idx, int ElementSize);

#define AC_THREADED

#endif
