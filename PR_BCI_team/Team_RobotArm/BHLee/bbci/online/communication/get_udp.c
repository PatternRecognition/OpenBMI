#include "udp.h"
#include <mex.h>
#include <math.h>

#ifdef _WIN32
#define trunc(x) floor(x)
#define round(x) floor(x + .5)
#endif
/*
 * Global data
 */
int connected = 0;
int udp_socket;
int udp_port;
char udp_hostName[100];
struct sockaddr_in udp_sa;

/************************************************************/

/*
 * init - initialize a socket and translate the name
 */
void init(char *hostname, int portnumber)
{
  if( (udp_socket = udp_createreadsocket(hostname, portnumber)) == -1) {
    mexErrMsgTxt("Couldn't create the socket");
    return;
  }

  udp_port = portnumber;
 	memcpy(&(udp_hostName[0]),hostname,100);
  connected = 1;
}

/*
 * send - send data to the socket
 */
double *getdata(int *n, int *m, double timeout)
{
  int header[3] = { 0, 0, 0 };
  int *packet;
  int len;  /* number of doubles */
  int size; /* size of packet */
  double *array;
  time_t tv_sec;
  suseconds_t tv_usec;
  int nRead;
  
  if(!connected) return 0;

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
    if( udp_waitread(udp_socket, tv_sec, tv_usec) == 0) {
      /* time out */
      return 0;
    }
    
    /* read first few bytes */
    nRead = udp_read(udp_socket, header, 3*sizeof(int), MSG_PEEK, NULL);
    
  /*  printf("Read header: %08x, n = %d, m = %d\n",
	   header[0], header[1], header[2]);*/

    if (header[0] != 0x0b411510) {
      mexWarnMsgTxt("Dropping udp packet with unknown header!");
      udp_read(udp_socket, header, 3*sizeof(int), 0, NULL);
      continue;
    }

    /* get length of packet */
    *n = header[1];
    *m = header[2];
    len = *n * *m;
    size = 3*sizeof(int) + len*sizeof(double);

    /* read packet */
    packet = malloc(size);
    udp_read(udp_socket, packet, size, 0, NULL);

    /* allocate return array and copy data */
    array = malloc(len*sizeof(double));
    memcpy(array, packet + 3, len*sizeof(double));

    /* cleanup and return */
    free(packet);
  
    return array;
  }
}

void cleanup()
{
  if (connected) {
    close(udp_socket);
    connected = 0;
  }
}

/************************************************************/

/*
 * We have three different modes:
 * 
 * open a connection:  get_udp(hostname, port)
 * send data:          data = get_udp([timeout in secs])
 * close connection:   get_udp
 */
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  if (nrhs >= 2) {
    static char hostname[1024];
    int portnumber;

    /* Check argument types */
    mxAssert(mxIsChar(prhs[0]), "Hostname must be a string.");
    mxAssert(mxIsNumeric(prhs[1]), "Portname must be a scalar.");

    /* Get the arguments */
    mxGetString(prhs[0], hostname, 1024);
    portnumber = mxGetScalar(prhs[1]);

	if(connected) {
		if(memcmp(&(udp_hostName[0]),hostname,100) == 0 && udp_port == portnumber) {
		  mexWarnMsgTxt("New socket is the same as the old one.");
		  return;
		}
      mxErrMsgTxt("Already connected");
	} else {
      init(hostname, portnumber);
      
      connected = 1;
    }
  }
  if (nrhs == 1 && mxIsChar(prhs[0])) {
    if(!connected)
      mxErrMsgTxt("get_udp: Not connected!\n");
    else {
      cleanup();
    }
  }
  else if (nrhs == 0 || nrhs == 1) {
    if(!connected)
      mxErrMsgTxt("get_udp: Not connected!\n");
    else {
      int n, m;
      double *array;
      double timeout = -1;
      
      if (nrhs == 1) { 
	mxAssert(mxIsNumeric(prhs[0]), "Timeout must be a scalar.");
	timeout = mxGetScalar(prhs[0]);
      }

      array = getdata(&n, &m, timeout);

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
}
