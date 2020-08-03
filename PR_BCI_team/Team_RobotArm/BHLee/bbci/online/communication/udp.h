/*
 * udp_dasher - a minimal udp layer for the dasher control
 *
 * (c) 2005 by Mikio Braun, modified by kraulem
 */

#ifndef UDP_H
#define UDP_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#define close(s) closesocket(s)
/*typedef long time_t;*/
typedef long suseconds_t;
#else /* UNIX */
#include <netdb.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/time.h>
#endif

/*
 * In general, all functions print error messages to stderr and return
 * -1 on error.
 */

/*
 * udp sockets for reading.
 *
 * On construction with udp_creatreadsocket() you specify the port to
 * which the socket is bound. You can do blocking reads with
 * udp_read(), or wait for data to be available with timeout with
 * udp_waitread()
 */
extern int udp_createreadsocket(const char *name, int port);
extern int udp_read(int s, void *buffer, int size, int flags,
		    struct in_addr *sender_addr);
extern int udp_waitread(int s, time_t tv_sec, suseconds_t tv_usec);

/*
 * udp sockets for writing.
 *
 * You create a socket for sending with udp_createsendsocket(). Since
 * the target address is passed to each send, there are no
 * arguments. udp_sockaddrbyname computes a sockaddr from a hostname
 * and a portnumber. udp_send() sends a fixed data set. 
 */ 
extern int udp_createsendsocket();
extern int udp_sockaddrbyname(struct sockaddr_in *sa, const char *s, int port);
extern int udp_send_dasher(int s, void *buffer, int size, struct sockaddr_in *sa);

#endif
