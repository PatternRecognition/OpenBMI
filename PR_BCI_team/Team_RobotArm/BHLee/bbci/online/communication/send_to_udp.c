#include "udp.h"
#include <mex.h>
#include <math.h>

/*
 * Global data
 */

#define MAX_CONNECTIONS    64

#define IS_HANDLE(a)	(mxIsUint16(a))
#define GET_HANDLE(a)	(*(unsigned short*)mxGetPr(a))

int findHandle();
mxArray *makeHandle(unsigned short h);
void printstatus();

int initialized = 0;

struct connection {
  int connected;
  int udp_socket;
  struct sockaddr_in udp_sa;
} connections[MAX_CONNECTIONS];

/************************************************************/

/*
 * init - initialize a socket and translate the name
 */
int init(char *hostname, int portnumber)
{
  int handle = findHandle();

  if( handle != -1) {
    struct connection *c = &connections[handle];
    if( (c->udp_socket = udp_createsendsocket()) == -1) {
      mexErrMsgTxt("Couldn't create the socket");
      return -1;
    }
    
    if( udp_sockaddrbyname(&c->udp_sa, hostname, portnumber) == -1) {
      mxErrMsgTxt("Couldn't resovle hostname");
      return -1;
    }
    
    c->connected = 1;
  }
  return handle;
}

/*
 * send - send data to the socket
 */
void senddata(int handle, int n, int m, double *array)
{
  struct connection *c = &connections[handle];
  int len, size, *packet, nSent;
  
  if(handle == -1 || !c->connected) return;

  /* construct the packet */
  len = n*m; 
  size = 3 * sizeof(int) + len * sizeof(double) ;
  packet = malloc( size );

  packet[0] = 0x0b411510;
  packet[1] = n;
  packet[2] = m;
  memcpy(packet + 3, array, len*sizeof(double));

  nSent = udp_send(c->udp_socket, packet, size, &c->udp_sa);
  if (nSent == -1)
    printf("send_to_udp: Error sending data\n");

  free(packet);
}

void cleanup(int handle)
{
  struct connection *c = &connections[handle];

  if (handle != -1 && c->connected) {
    close(c->udp_socket);
    c->connected = 0;
  }
}

/************************************************************/

/*
 * We have three different modes:
 * 
 * open a connection:  handle = send_to_udp(hostname, port)
 * send data:          send_to_udp(handle, data)
 * close connection:   send_to_udp(handle, 'close')
 */
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  /* initialize arrays if necessary */
  if(!initialized) {
    int i;
    for(i = 0; i < MAX_CONNECTIONS; i++)
      connections[i].connected = 0;
    initialized = 1;
  }

  /* open a connection */
  if (nrhs == 2 & mxIsChar(prhs[0])) {
    static char hostname[1024];
    int portnumber;
    int handle;

    /* Check argument types */
    mxAssert(mxIsChar(prhs[0]), "Hostname must be a string.");
    mxAssert(mxIsNumeric(prhs[1]), "Portname must be a scalar.");
    
    /* Get the arguments */
    mxGetString(prhs[0], hostname, 1024);
    portnumber = mxGetScalar(prhs[1]);

    handle = init(hostname, portnumber);

    /* if everythings okay, return the handle, otherwise an empty matrix */
    if(handle != -1) {
      plhs[0] = makeHandle(handle);
    }
    else {
      plhs[0] = mxCreateDoubleMatrix(0,0, mxREAL);
    }

  }
  else if (nrhs == 2 && IS_HANDLE(prhs[0]) && mxIsChar(prhs[1])) {
    int handle = GET_HANDLE(prhs[0]);
    if(!connections[handle].connected)
      mxErrMsgTxt("send_to_udp: Not connected");
    else {
      cleanup(handle);
    }
  }
  else if (nrhs == 2 && IS_HANDLE(prhs[0])) {
    int handle = GET_HANDLE(prhs[0]);
    if(!connections[handle].connected)
      mxErrMsgTxt("send_to_udp: Not connected");
    else {
      int n, m;
      double *array;
      
      /* Check argument types */
      mxAssert(mxIsDouble(prhs[1]), "Can only send double arrays");
      
      n = mxGetN(prhs[1]);
      m = mxGetM(prhs[1]);
      array = mxGetPr(prhs[1]);
      
      senddata(handle, n, m, array);
    }
  }
  else {
    mxErrMsgTxt("send_to_udp: call with arguments (hostname, port), (handle, data), or (data)");
  }
}

/************************************************************
 *
 * Auxillary functions
 *
 *************************************************************/

mxArray *makeHandle(unsigned short i)
{
  int dims[2] = { 1, 1 };
  mxArray *a = mxCreateNumericArray(2, dims, mxUINT16_CLASS, mxREAL);
  *(unsigned short*)mxGetPr(a) = i;
  return a;
}


int findHandle()
{
  int i;
  int handle = -1; 
  for(i = 0; i < MAX_CONNECTIONS; i++) {
    if( !connections[i].connected ) {
      handle = i;
      break;
    } 
  }
  if( handle == -1 )
    mxErrMsgTxt("send_to_udp: No handles left!\n");

  return handle;
}

void printstatus()
{
  int i;
  for(i = 0; i < MAX_CONNECTIONS; i++) {
    struct connection *c = &connections[i];
    if(c->connected)
      fprintf(stderr, "%d connected\n", i);
  }
}
