/*
  headerfifo.h

  A hand-coded fifo list to store the messages received from the
  brainvision server.

  Written by Mikio Braun, mikio@first.fhg.de

  (c) Fraunhofer FIRST.IDA 2005
*/

#ifndef HEADER_FIFO_H
#define HEADER_FIFO_H

#include "myRDA.h"

/* fifo data structures */
typedef struct RDA_MessageHeader RMH;

struct headerFIFONode
{
        struct headerFIFONode *prev, *next;
        RMH *payload;
};

struct headerFIFO
{
    int length;
    struct headerFIFONode *first;
    struct headerFIFONode *last;
};

extern struct headerFIFO fifo;

#define FIFOMAXLEN	1024

extern void headerFIFOpush(struct headerFIFO *p, RMH *h);
extern RMH *headerFIFOpop(struct headerFIFO *p);

#endif
