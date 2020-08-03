#ifndef AMPLIFIERCONFIG_H
#define AMPLIFIERCONFIG_H
#include <string>
#include <vector>
#include "EEGHeader.h"
#include "Globals.h"

using namespace std;

/**
	This class defines the configuration object used by classes inherited from Amplifier
	This object is passed to the EEGAccess class and contains the information required to setup
	the appropriate amplifier for recording
*/
class AmplifierConfig
{


public:
	

	AmplifierConfig();
	AmplifierConfig( vector<string> channelNames, int sampleRate, int bufferSize, AmpType ampType, string inFile="", string outFile="");
	~AmplifierConfig(){};

	/**
		Return the number of channels to be used by the amp
	*/
	int getNumChannels();

	/**
		Set the names of the channels to be used by the amp
		@param a vector of strings containing the channel names
	*/
	void setChannelNames( vector<string> channelNames);

	/**
		return the name of a channel at a given index
	*/
	string getChannelName(int index);

	/**
		return the buffer size.
	*/
	int getBufferSize();

	/**
		set buffer size
		@param buffer size
	*/
	void setBufferSize(int bufferSize);

	/**
		get the sample rate used by the amp in Hz
	*/
	int getSampleRate();

	/**
		set the sample rate used by the amp in Hz
		@param sample rate
	*/
	void setSampleRate(int sampleRate);

	/**
		get the filename where recorded eeg data will be saved
	*/
	string getOutFileName();

	/**
		set the filename where recorded eeg data will be saved
		@param the file path
	*/
	void setOutFileName(string outFile);

	/**
		get the filename where the eeg data will be read from (for the Amplifier_File class)
	*/
	string getInFileName();

	/**
		set the filename where the eeg data will be read from (for the Amplifier_File class)
		@param the file path
	*/
	void setInFileName(string inFile);

	/**
		Return the amp type to be used
	*/
	AmpType getAmpType();

	/**
		Set the amp type to be used
		@param the amp type
	*/
	void setAmpType(AmpType ampType);

	/**
		Set the header object
	*/
	void setHeader(EEGHeader* header);

	/**
		get the header object
	*/
	EEGHeader* getHeader();


	/**
		get the file formate
		@return FileFormat to use
	*/
	FileFormat getFileFormat();

	/**
		set the file format
		@param the file format

	*/
	void setFileFormat(FileFormat format);

	/**
		serialise the config data to an iostream
		@param the iostream to serialise to
	*/
	void serialise(iostream &stream);

	/**
		deserialis the config data from an iostream
		@param the iostream to deserialise from
	*/
	void deserialise(iostream &stream);

	/**
		print the contents of the config object to the terminal
	*/
	void print();



private:
	/// the header file associated with this config
	EEGHeader *m_header;
	/// the filename of the eeg data file to read data from
	string m_inFile;
	/// the filename of the eeg data file to write to
	string m_outFile;
	/// the type of amplifier
	AmpType m_ampType;
	/// the buffersize to be used
	int m_bufferSize;
};


#endif
