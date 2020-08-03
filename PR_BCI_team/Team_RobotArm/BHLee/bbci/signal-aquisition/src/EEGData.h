#ifndef EEGDATA_H
#define EEGDATA_H

#include <string>
#include <vector>
#include <fstream>
#include <boost/thread/thread.hpp>
#include "Globals.h"
#include "EEGHeader.h"
#include "EEGMarker.h"

using namespace std;

/**
 An object used to store eeg data
*/
class EEGData
{



private:
	/// a 2D vector containing the eeg data samples
	/// dimensions = numSamples x numChannels
	/// used as a FIFO queue
	vector< vector <double> > m_samples;
	/// number of channels of data 
	int m_numChannels;
	/// io mutex
	boost::mutex m_io_mutex;
	/// whether or not m_samples has an extra column for markers
	bool m_hasMarkers;
	/// the position of the start of this eegdata object 
	int m_dataStartPos;
	


public:
	/**
		enum of the types of amplifiers currently available
	*/
	
	EEGData( );
	EEGData( int startPos);
	~EEGData(){};



	/**
		@return whether or not this eegdata includes markers
	*/
	bool hasMarkers();

	/**
		@param does eeg data include a marker channel
	*/
	void includeMarkers(bool includeMarkers);
	
		
	/**
		@return the StartPos of data
	*/
	int getStartPos();

	/**
		set the StartPos of data
		@param the StartPos
	*/
	void setStartPos(int startPos);


	/**
		@return the number of channels of data
	*/
	int getNumChannels();

	/**
		set the number of channels of data
		@param the number of channels
	*/
	void setNumChannels(int numChannels);

	/**
		@return the number of samples currently stored
	*/
	int getNumSamples();

	/**
		add a new sample
		@param a vector of doubles
	*/
	void addSample(vector<double> sample);

	/**
		add a number of new samples
		@param a vector of vectors of doubles
	*/
	void addSamples(vector< vector<double> > samples);

	/**
		get the next sample from m_samples. FIFO queue
		returned sample is removed from the eegdata object
		@return next eeg sample data
	*/
	vector<double> getNextSample();

	/**
		get the next n samples from m_samples.
		returned samples are removed from the eegdata object
		@param the number of samples to return, if numSamples = 0, return all available sample
		@return eeg sample data
	*/
	vector< vector<double> > getSamples(int numSamples);

	/**
		remove markers from the data and return them
	*/
	EEGMarker* extractMarkers();

	/**
		add marker data/channel
		@param markers that can be added
	*/
	void addMarkers(EEGMarker* markers);

	/**
		print the object details to the terminal
	*/
	void print();

	/**
		serialise the object to an iostream
		@param the iostream to serialise to
	*/
	void serialise(iostream &stream, EEGHeader *header);

	/**
		deserialise the object from an iostream
		@param the iostream to deserialise from
	*/
	void deserialise(iostream &stream, EEGHeader *header, int maxRead);

	/**
		@return a copy of this eegdata object
	*/
	EEGData* copy();


};

#endif


