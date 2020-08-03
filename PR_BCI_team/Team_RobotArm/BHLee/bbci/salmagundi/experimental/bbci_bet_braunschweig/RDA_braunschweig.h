#ifndef _INC_RECORDERRDA
#define _INC_RECORDERRDA
/* // based on MODULE: RDA.h
   //: written by: Henning Nordholz
   //+       date: 14-Nov-00
   //+ 
   //+ Adapted by Martin Oehler
*/

#pragma pack(1)
#ifndef ULONG
typedef unsigned long ULONG;
#endif


/* All numbers are sent in little endian format. */

/* Unique identifier for messages sent to clients
   {4358458E-C996-4C86-AF4A-98BBF6C91450}
   As byte array (16 bytes): 8E45584396C9864CAF4A98BBF6C91450
   0x4358458e, 0xc996, 0x4c86, 0xaf, 0x4a, 0x98, 0xbb, 0xf6, 0xc9, 0x14, 0x50);
   DEFINE_GUID(GUID_RDAHeader,
   0x4358458e, 0xc996, 0x4c86, 0xaf, 0x4a, 0x98, 0xbb, 0xf6, 0xc9, 0x14, 0x50);
*/

struct RDA_Marker
/*; A single marker in the marker array of RDA_MessageData */
{
	ULONG				nSize;				/* Size of this marker. */
	ULONG				nPosition;			/* Relative position in the data block. */
	ULONG				nPoints;			/* Number of points of this marker */
	long				nChannel;			/* Associated channel number (-1 = all channels). */
	char				sTypeDesc[1];		/* Type, description in ASCII delimited by '\0'. */
};


struct RDA_MessageHeader
/*; Message header */
{
	unsigned char guid[16];		/* Always GUID_RDAHeader */
	ULONG nSize; 	/* Size of the message block in bytes including this header */
	ULONG nType;	/* Message type. */
};


/* **** Messages sent by the RDA server to the clients. **** */
struct RDA_MessageStart
/*; Setup / Start infos, Header -> nType = 1 */
{
	unsigned char guid[16];		/* Always GUID_RDAHeader */
	ULONG nSize; 	/* Size of the message block in bytes including this header */
	ULONG nType;	/* Message type. */
	ULONG				nChannels;			/* Number of channels */
	double				dSamplingInterval;	/* Sampling interval in microseconds */
	double				dResolutions[1];	/* Array of channel resolutions -> double dResolutions[nChannels] */
											/* coded in microvolts. i.e. RealValue = resolution * A/D value */
	char 				sChannelNames[1];	/* Channel names delimited by '\0'. The real size is  */
											/* larger than 1. */
};


struct RDA_MessageStop
/*; Data acquisition has been stopped. /* Header -> nType = 3 */
{
	unsigned char guid[16];		/* Always GUID_RDAHeader */
	ULONG nSize; 	/* Size of the message block in bytes including this header */
	ULONG nType;	/* Message type. */
};

struct RDA_MessageData32
/*; Block of 32-bit floating point data, Header -> nType = 4, sent only from port 51236 */
{
	unsigned char guid[16];		/* Always GUID_RDAHeader */
	ULONG nSize; 	/* Size of the message block in bytes including this header */
	ULONG nType;	/* Message type. */
	ULONG				nBlock;				/* Block number, i.e. acquired blocks since acquisition started. */
	ULONG				nPoints;			/* Number of data points in this block */
	ULONG				nMarkers;			/* Number of markers in this data block */
	float				fData[1];			/* Data array -> float fData[nChannels * nPoints], multiplexed */
	struct RDA_Marker			Markers[1];			/* Array of markers -> RDA_Marker Markers[nMarkers] */
};


/* **** End Messages sent by the RDA server to the clients. **** */

#pragma pack()

#endif /*_INC_RECORDERRDA */
