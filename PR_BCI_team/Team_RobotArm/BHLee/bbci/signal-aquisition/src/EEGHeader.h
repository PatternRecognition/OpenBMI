#ifndef EEGHEADER_H
#define EEGHEADER_H

#include <string>
#include <vector>
#include <fstream>
#include "Globals.h"

using namespace std;

/**
	the header file class contains information about the eeg data we will/have recorded

*/
class EEGHeader
{



private:
	// the names of the channels in this recording
	vector<string> m_channelNames;
	// the sample rate the recording was made at
	int m_sampleRate;
	// endian format
	char m_endian;
	// binary format 1 = int16 2=int32 3=float 4=double
	BinaryDataFormat m_binaryDataFormat;
	// header format
	FileFormat m_FileFormat;
	// name of the data file, not used for standard header format
	string m_DataFileName;


	/**
		standard initialisation 
	*/
	void init();
	
	/**
		serialise the object ot an iostream in standard format
		@param the iostream to serialise to
	*/
	void serialise_standard(iostream &stream);

	
	/**
		serialise the object ot an iostream in brain vision format
		@param the iostream to serialise to
	*/
	void serialise_bv(iostream &stream);

	/**
		deserialise the object from an iostream in standard format
		@param the iostream to deserialise from
	*/
	void deserialise_standard(iostream &stream);

	/**
		deserialise the object from an iostream in brain vision format
		@param the iostream to deserialise from
	*/
	void deserialise_bv(iostream &stream);




public:
	EEGHeader();

	/**
	 a constructor for the EEGHeader class that also initialises the channel names and sample rate
	 @param a vector of strings containing channel names
	 @param sample rate
	*/
	EEGHeader( vector<string> channelNames, int sampleRate);

	~EEGHeader(){};

	/**
		@return the number of channels in this recording
	*/
	int getNumChannels();

	/**
		set the names of the channels
		@param a vector of strings containing channel names
	*/
	void setChannelNames( vector<string> channelNames);

	/**
		@param the index of the channel required
		@return the name of the channel at the given index

	*/
	string getChannelName(int index);

	/**
		@return the sample rate of the data
	*/
	int getSampleRate();

	/**
		set the sample rate of the data
		@param sample rate
	*/
	void setSampleRate(int sampleRate);

	/**
		print the object details to the terminal
	*/
	void print();

	/**
		serialise the object ot an iostream
		@param the iostream to serialise to
	*/
	void serialise(iostream &stream);

	/**
		deserialise the object form an iostream
		@param the iostream to deserialise from
	*/
	void deserialise(iostream &stream);

	/**
		returns endian format of data
		@return 'l' or 'b'
	*/
	char getEndian();

	/**
		returns binary format of data
		@return 1 = int16 2=int32 3=float 4=double
	*/
	BinaryDataFormat getBinaryDataFormat();

	/**
		set binary format of data file
		
	*/
	void setBinaryDataFormat(BinaryDataFormat format);

	/**
		returns the file format enum for the header
		@return 1= BV 2 = std
	*/
	FileFormat getFileFormat();

	/**
		set the file format for the header
	*/
	void setFileFormat(FileFormat);

	/**
		set the name of the data file
	*/
	void setDataFileName(string name);

	/**
		GET the name of the data file
		@return the name of the data file
	*/
	string getDataFileName();
};

#endif


