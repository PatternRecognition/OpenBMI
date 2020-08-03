#include "EEGData.h"
#include "EEGMarker.h"
static void swap16(char *b)
{
    char temp = b[0]; b[0]=b[1]; b[1]=temp;   
}

static void swap32(char *b)
{
    char temp;
    temp = b[0]; b[0]=b[3]; b[3]=temp;
    temp = b[1]; b[1]=b[2]; b[1]=temp;    
}


/*************************************************************
 *
 * Checks the endian format of this machine
 *
 *************************************************************/
static char endian() {
    int i = 1;
    char *p = (char *)&i;

    if (p[0] == 1)
        return 'l'; /*least important byte is first byte  */
    else
        return 'b'; /*least important byte is last byte  */
}
  

