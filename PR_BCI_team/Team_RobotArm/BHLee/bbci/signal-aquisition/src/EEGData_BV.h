#ifndef EEGDATA_BV_H
#define EEGDATA_BV_H

/*
 * FORWARD DECLARATIONS
 */ 

#include <string>
#include <stdio.h>
using namespace std;
class EEGData_BV
{


private:
		
	/*char *FORMAT_FIELD;
	const char *FS_FIELD;
	const char *N_CHANS_FIELD;
	const char *N_POINTS_FIELD;
	const char *SCALE_FIELD;
	const char *ENDIAN_FIELD;
	const char *CHAN_ID_X_FIELD;
	const char *FILT_A_FIELD;
	const char *FILT_B_FIELD;
	const char *FILT_SUBSAMPLE_FIELD;
	const char *DATA;
	const char *DATA_POS;*/

	/* hdr data values */
	double rawDataSamplingRate;
	int rawDataChannelCount;
	int binaryDataFormat;
	int rawElementSize;
	double *rawDataScale;
	char rawDataEndian;
	int rawDataPoints;

	/* opt non optional values */
	double optChannelSelect;
	int optChannelSelectCount;
	double optSamplingRate;

	/* the handle for the eeg-file */
	FILE *eegFile;

	

	int lag; /* the difference between the sampling rate of the raw data and
				  and the sampling rate of the requested data */

	/* the positions of the samples when we write in a matrix*/
	double dataPtr;
	int dataPtrSize;         /* the number of rows in the data */
	int dataStart;           /* the position of the first sample in the data*/
	int dataEnd;             /* the position of the last sample in the data*/
	int fileStart;           /* the first position of the data in the file*/ 
	int fileEnd;             /* the last position of the data in the file*/

	
public:
	
	
	EEGData_BV();
	~EEGData_BV(){};

	int file_length(FILE *f);

	void swap16(char *b);

	void swap32(char *b);

	char endian();

	void rbv_init(int nrhs, string filename );

	bool rbv_readDataBlock(double *dataBlock, void* dataBuffer, int channelCount, bool swap);

	

	void rbv_cleanup();

	void rbv_assert(bool aValue,const char* text);

};
#endif