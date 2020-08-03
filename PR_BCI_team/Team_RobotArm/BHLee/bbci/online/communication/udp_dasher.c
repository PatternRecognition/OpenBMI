#include "udp_dasher.h"

#define ec_neg1( expr ) if ( (expr) == - 1) { \
   perror(#expr " failed"); \
   return -1; \
   }

int udp_sockaddrbyname(struct sockaddr_in *sa, const char *s, int port)
{
  struct hostent *he;

  he = gethostbyname(s);
  if(he == NULL) {
#ifdef _WIN32
    printf("Host %s unknown\n", s);
#else
    printf("Host %s unknown: %s\n", s, hstrerror(h_errno));
#endif
    return -1;
  }

  sa->sin_family = AF_INET;
  sa->sin_port = htons(port);
  sa->sin_addr = *((struct in_addr*)(he->h_addr_list[0]));  

  return 0;
}

/*
 * create a udp socket for reading
 *
 * name has to refer to the local machine
 */
int udp_createreadsocket(const char *name, int port)
{
  int s;
  struct sockaddr_in sa;

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
  
  /* create socket */
  ec_neg1( s = socket(AF_INET, SOCK_DGRAM, 0) );
    
  /* resolve host name */
  ec_neg1( udp_sockaddrbyname(&sa, name, port) );

  /* bind socket */
  ec_neg1( bind(s, (struct sockaddr *)&sa, sizeof(sa)) );

  return s;
}


/*
 * read a packet from the udp socket
 *
 * if sender_addr != NULL, it will contain the ip-address of the
 * sender afterwards.
 */
int udp_read(int s, void *buffer, int size, int flags, 
	     struct in_addr *sender_addr)
{
  int bytesread;

  if( sender_addr != NULL) {
    struct sockaddr sender_sa;
    struct sockaddr_in *si;
    int sa_len = sizeof(struct sockaddr);  

    ec_neg1( bytesread = recvfrom(s, buffer, size, flags, 
				  (struct sockaddr*)&sender_sa, &sa_len) );

    si = (struct sockaddr_in *) &sender_sa;
    *sender_addr = si->sin_addr;
  }
  else {
    ec_neg1( bytesread = recvfrom(s, buffer, size, 
				  flags, NULL, NULL) );
  }
  return bytesread;
}


/*
 * wait for a read socket to become ready
 *
 * if tv_sec >= 0, a timeout is installed. If timed out, 0 is
 * returned, 1 if data is ready.
 */
int udp_waitread(int s, time_t tv_sec, suseconds_t tv_usec)
{
  struct timeval tv = { tv_sec, tv_usec };
  int ready;

  fd_set set;

  FD_ZERO(&set);
  FD_SET(s, &set);

  if (tv_sec >= 0) {
    ec_neg1( ready = select(s + 1, &set, NULL, NULL, &tv) );
  }
  else {
    ec_neg1( ready = select(s + 1, &set, NULL, NULL, NULL) );
  }
  return ready;
}

/*
 * create a udp socket for sending
 */
int udp_createsendsocket()
{
  int s;
  
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
  
  ec_neg1( s = socket(AF_INET, SOCK_DGRAM, 0) );
  return s;
}


/*
 * send packages to a socketaddress
 */
int udp_send_dasher(int s, void *buffer, int size, struct sockaddr_in *sa)
{
  ec_neg1( sendto(s, buffer, size, 0, 
		  (struct sockaddr*)sa, sizeof(*sa)) );

}
