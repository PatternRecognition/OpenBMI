	
#ifndef EEGMARKER_H
#define EEGMARKER_H

#include <string>
#include <vector>
#include <fstream>
#include <boost/thread/thread.hpp>

#include "EEGHeader.h"

using namespace std;

/**
 An object used to store eeg markers
*/
class EEGMarker
{

public:
	struct marker{int pos; string type; string val;};

private:

	
	/// a 2D vector containing marker samples
	/// dimensions = numSamples x 2 (column 1 is stmulus markers, 2 is response markers)
	/// used as a fifo queue
	vector< marker > m_samples;
	/// io mutex
	boost::mutex m_io_mutex;

	///return a padded value for the int as string for marker.val
	string padStimVal(int val);

	/**
		enum of the types of amplifiers currently available
	*/
	
public:
	
	EEGMarker();
	~EEGMarker(){};

	/**
		@return the number of samples currently stored
	*/
	int getNumSamples();

	/**
		add a new sample
		@param EEGMarker::maker
	*/
	void addSample(marker sample);

	/**
		add a new sample
		@param int from eegdata channel
		@param int of the data position
	*/
	void addSample(int val, int dataPos);

	

	/**
		add a number of new samples
		@param a vector of vectors of ints
	*/
	void addSamples(vector<marker> samples);

	

	/**
		get the next sample from m_samples. FIFO queue
		returned sample is removed from the eegmarker object
		@return next eeg sample marker
	*/
	marker getNextSample();


	/**
		get the next sample from m_samples. FIFO queue
		returned sample is removed from the eegmarker object
		@return next eeg sample marker as int for eeg channel
	*/
	int getNextSampleAsInt();

	/**
		get the next n samples from m_samples.
		returned samples are removed from the eegmarker object
		@param the number of samples to return, if numSamples = 0, return all available sample
		@return eeg sample marker
	*/
	vector< marker > getSamples(int numSamples);

	/**
		return next marker in queu without removing it 
	*/
	marker peek();
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
	void deserialise(iostream &stream);

	/**
		@return a copy of this eegmarker object
	*/
	EEGMarker* copy();

};
#endif