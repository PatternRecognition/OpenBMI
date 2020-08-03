/*
  headerfifo.c

*/

#include <stdlib.h>
#include "headerfifo.h"

struct headerFIFO fifo = { 0, 0, 0 };

void headerFIFOpush(struct headerFIFO *p, RMH *h)
{
  if(FIFOMAXLEN > 0 && p->length > FIFOMAXLEN) {
    /* dropping a packet! */
    free(h);
  }
  else {
    struct headerFIFONode *n = malloc(sizeof(struct headerFIFONode));
    n->prev = 0;
    n->next = 0;
    n->payload = h;

    /* 1st case: list is empty */
    if (!p->first && !p->last) {
        p->first = n;
        p->last = n;
    }
    else {
      /* 2nd case: at least one element in list -> everything is easy. */
        p->first->prev = n;
        n->next = p->first;
        p->first = n;
    }

    p->length++;
  }
}



RMH *headerFIFOpop(struct headerFIFO *p)
{
    RMH *h = 0;
    struct headerFIFONode *n = p->last;

    /* case 1: empty fifo */
    if (!n) return 0;

    /* case 2: only one element */
    else if(p->first == n) {
        h = n->payload;
        free(n);
        p->first = 0;
        p->last = 0;
    }
    /* case 3: more than one element */
    else {
        h = n->payload;
        p->last = n->prev;
        n->prev->next = 0;
        free(n);
    }

    p->length--;

    return h;
}
