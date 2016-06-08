/* MODULE: RDA.h
   written by: Henning Nordholz date: 14-Nov-00
 
   Description:
     Vision Recorder 
     Remote Data Access (RDA) structs and constants

   Adapted by Guido Dornhege, Mikio Braun
*/

#ifndef MY_RDA_H
#define MY_RDA_H

typedef unsigned long ULONG;
 
/*
#endif

*/
/* All numbers are sent in little endian format. */

/* Unique identifier for messages sent to clients
   {4358458E-C996-4C86-AF4A-98BBF6C91450}
   As byte array (16 bytes): 8E45584396C9864CAF4A98BBF6C91450
   DEFINE_GUID(GUID_RDAHeader,
   0x4358458e, 0xc996, 0x4c86, 0xaf, 0x4a, 0x98, 0xbb, 0xf6, 0xc9, 0x14, 0x50);
*/

#pragma pack(push)

/* A single marker in the marker array of RDA_MessageData */
#pragma pack(1)
struct RDA_Marker
{
  ULONG  nSize;          /* Size of this struct. */
  ULONG  nPosition;      /* Relative position in the data block. */
  ULONG  nPoints;        /* Number of points of this marker */
  long   nChannel;       /* Associated channel number (-1 = all). */
  char   sTypeDesc[1];   /* Type, description in ASCII delimited by '\0'. */
};


#pragma pack(1)
struct RDA_MessageHeader
{
  unsigned char  guid[16];    /* Always GUID_RDAHeader */
  ULONG  nSize;               /* Size of this struct */
  ULONG  nType;               /* Message type. */
};

/* Setup / Start infos, Header -> nType = 1 */
#pragma pack(1)
struct RDA_MessageStart
{
  unsigned char  guid[16];     /* Always GUID_RDAHeader */
  ULONG  nSize;                /* Size of this struct */
  ULONG  nType;                /* Message type. */
  ULONG  nChannels;
  double dSamplingInterval;    /* Sampling interval in microseconds */
  double dResolutions[1];  
/* Array of channel resolutions -> double dResolutions[nChannels] */
/* coded in microvolts. i.e. RealValue = resolution * A/D value   */
  char   sChannelNames[1];  
/* Channel names delimited by '\0'. The real size is larger than 1. */
};


/* Block of data, Header -> nType = 2 */
#pragma pack(1)
struct RDA_MessageData
{
  unsigned char  guid[16];   /* Always GUID_RDAHeader */
  ULONG  nSize;              /* Size of this struct */
  ULONG  nType;              /* Message type. */
  ULONG  nBlock;        
/* Block number, i.e. acquired blocks since acquisition started. */
  ULONG  nPoints;            /* Number of data points in this block */
  ULONG  nMarkers;           /* Number of markers in this data block */
  short  nData[1];
/* Data array -> short nData[nChannels * nPoints], multiplexed */
  struct RDA_Marker   Markers[1];
/* Array of markers -> RDA_Marker Markers[nMarkers] */
};


/* Data acquisition has been stopped. // Header -> nType = 3 */
#pragma pack(1)
struct RDA_MessageStop
{
  unsigned char  guid[16];    /* Always GUID_RDAHeader */
  ULONG nSize;                /* Size of this struct */
  ULONG nType;                /* Message type. */
};

#pragma pack(pop)

#endif

