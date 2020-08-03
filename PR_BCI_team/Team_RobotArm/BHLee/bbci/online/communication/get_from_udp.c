#include "udp.h"
#include <mex.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#define trunc(x) floor(x)
#define round(x) floor(x + .5)
#endif
/*
 * Global data
 */

#define MAX_CONNECTIONS    64

#define IS_HANDLE(a)	(mxIsUint16(a))
#define GET_HANDLE(a)	(*(unsigned short*)mxGetPr(a))

int findHandle();
mxArray *makeHandle(unsigned short h);

int initialized = 0;

struct connection {
  int connected;
  int udp_socket;
  int udp_port;
  char udp_hostName[100];
  /*struct sockaddr_in udp_sa;*/
} connections[MAX_CONNECTIONS];

/************************************************************/

/*
 * init - initialize a socket and translate the name
 */
int init(char *hostname, int portnumber)
{
  int i;
  int handle = findHandle();

  if(handle != -1) {
    struct connection *c = &connections[handle];

    if( (c->udp_socket = udp_createreadsocket(hostname, portnumber)) == -1) {
      for(i = 0; i < 64;i++) {
        if(connections[i].connected) {
          if(memcmp(&(connections[i].udp_hostName[0]),hostname,100) == 0 && connections[i].udp_port == portnumber) {
            mexWarnMsgTxt("Couldn't create the socket using the old socket.");
            return i;
          }
        }
      }
      mexErrMsgTxt("Couldn't create the socket");
      return 0;
    }
    
  	c->udp_port = portnumber;
  	memcpy(&(c->udp_hostName[0]),hostname,100);
    c->connected = 1;
  }

  return handle;
}

/*
 * send - send data to the socket
 */
double *getdata(int handle, int *n, int *m, double timeout)
{
  struct connection *c = &connections[handle];
  
  int header[3] = { 0, 0, 0 };
  int *packet;
  int len;  /* number of doubles */
  int size; /* size of packet */
  double *array;
  time_t tv_sec;
  suseconds_t tv_usec;
  int nRead;
  
  if(!c->connected) return 0;

  /* calculate timeout */
  if (timeout < 0) {
    tv_sec = -1;
    tv_usec = 0;
  }
  else {
    tv_sec = (time_t)trunc(timeout);
    tv_usec = (suseconds_t)round((timeout - tv_sec)*1000000);
  }

  while(1) {
    /* wait for a packet to become available */
    if( udp_waitread(c->udp_socket, tv_sec, tv_usec) == 0) {
      /* time out */
      return 0;
    }
    
    /* read first few bytes */
    nRead = udp_read(c->udp_socket, header, 3*sizeof(int), MSG_PEEK, NULL);
    
  /*  printf("Read header: %08x, n = %d, m = %d\n",
	   header[0], header[1], header[2]);*/

    if (header[0] != 0x0b411510) {
      mexWarnMsgTxt("Dropping udp packet with unknown header!");
      udp_read(c->udp_socket, header, 3*sizeof(int), 0, NULL);
      continue;
    }

    /* get length of packet */
    *n = header[1];
    *m = header[2];
    len = *n * *m;
    size = 3*sizeof(int) + len*sizeof(double);

    /* read packet */
    packet = malloc(size);
    udp_read(c->udp_socket, packet, size, 0, NULL);

    /* allocate return array and copy data */
    array = malloc(len*sizeof(double));
    memcpy(array, packet + 3, len*sizeof(double));

    /* cleanup and return */
    free(packet);
  
    return array;
  }
}

void cleanup(int handle)
{
  struct connection *c = &connections[handle];
  if (c->connected) {
    close(c->udp_socket);
    c->connected = 0;
  }
}

/************************************************************/

/*
 * We have three different modes:
 * 
 * open a connection:  handle = get_from_udp(hostname, port)
 * get data:           data = get_from_udp(handle, [timeout in secs])
 * close connection:   get_from_udp('close')
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
  if (nrhs >= 2 && mxIsChar(prhs[0])) {
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

  /* get data */
  else if (IS_HANDLE(prhs[0]) &&
	   (nrhs == 1 || 
	    (nrhs == 2 && mxIsNumeric(prhs[1])))) {
    int handle = GET_HANDLE(prhs[0]);
    if(!connections[handle].connected)
      mxErrMsgTxt("get_from_udp: Not connected");
    else {
      int n, m;
      double *array;
      double timeout = -1;
      
      if (nrhs == 2) { 
	mxAssert(mxIsNumeric(prhs[1]), "Timeout must be a scalar");
	timeout = mxGetScalar(prhs[1]);
      }

      array = getdata(handle, &n, &m, timeout);

      if(array) {
	plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
	memcpy(mxGetPr(plhs[0]), array, n*m*sizeof(double));
	free(array);
      }
      else { /* timed out */
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
      }
    }
  }

  /* close */
  else if (nrhs == 2 && IS_HANDLE(prhs[0]) && mxIsChar(prhs[1])) {
    int handle = GET_HANDLE(prhs[0]);
    if(!connections[handle].connected)
      mxErrMsgTxt("get_from_udp: Not connected");
    else {
      cleanup(handle);
    }
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
    mexErrMsgTxt("get_from_udp: No handles left!\n");

  return handle;
}
