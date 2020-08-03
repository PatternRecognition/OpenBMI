#include "udp.h"
#include <mex.h>
#include <math.h>

/*
 * Global data
 */
int connected = 0;
int udp_socket;
struct sockaddr_in udp_sa;

/************************************************************/

/*
 * init - initialize a socket and translate the name
 */
void init(char *hostname, int portnumber)
{
  if( (udp_socket = udp_createsendsocket()) == -1) {
    mexErrMsgTxt("Couldn't create the socket");
    return;
  }

  if( udp_sockaddrbyname(&udp_sa, hostname, portnumber) == -1) {
    mxErrMsgTxt("Couldn't resovle hostname");
    return;
  }

  connected = 1;
}

/*
 * send - send data to the socket
 */
void senddata(int n, int m, double *array)
{
  int len, size, *packet, nSent;
  if(!connected) return;

  /* construct the packet */
  len = n*m; 
  size = 3 * sizeof(int) + len * sizeof(double) ;
  packet = malloc( size );

  packet[0] = 0x0b411510;
  packet[1] = n;
  packet[2] = m;
  memcpy(packet + 3, array, len*sizeof(double));

  nSent = udp_send(udp_socket, packet, size, &udp_sa);
  if (nSent == -1)
    printf("send_udp: Error sending data\n");

  free(packet);
}

void cleanup()
{
  if (connected)
    close(udp_socket);
}

/************************************************************/

/*
 * We have three different modes:
 * 
 * open a connection:  send_udp(hostname, port)
 * send data:          send_udp(data)
 * close connection:   send_udp
 */
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  if (nrhs == 2) {
    static char hostname[1024];
    int portnumber;

    /* Check argument types */
    mxAssert(mxIsChar(prhs[0]), "Hostname must be a string.");
    mxAssert(mxIsNumeric(prhs[1]), "Portname must be a scalar.");
    
    /* Get the arguments */
    mxGetString(prhs[0], hostname, 1024);
    portnumber = mxGetScalar(prhs[1]);

    init(hostname, portnumber);

    connected = 1;
  }
  else if (nrhs == 1) {
    if(!connected)
      mxErrMsgTxt("send_udp: Not connected!\n");
    else {
      int n, m;
      double *array;
      
      /* Check argument types */
      mxAssert(mxIsDouble(prhs[0]), "Can only send double arrays");
      
      n = mxGetN(prhs[0]);
      m = mxGetM(prhs[0]);
      array = mxGetPr(prhs[0]);
      
      senddata(n, m, array);
    }
  }
  else {
    if(!connected)
      mxErrMsgTxt("send_udp: Not connected!\n");
    else {
      cleanup();
    }
  }
}
