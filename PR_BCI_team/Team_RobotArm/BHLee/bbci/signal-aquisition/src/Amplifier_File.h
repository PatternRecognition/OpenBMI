#ifndef AMPLIFIER_FILE_H
#define AMPLIFIER_FILE_H

#include "Amplifier.h"
#include "EEGMarker.h"
#include <fstream>
#include <boost/thread/thread.hpp>



/**
	This class inherits from the Amplifier class
	It implements the Amplifier funcitonality for reading from saved EEG data files
	The amp then returns the data in real time as if the recording was being done live

*/
class Amplifier_File : public Amplifier
{

private:
	/// load marker data into this object
	EEGMarker m_markers;
	/// file to read the eeg data from
	fstream m_dataFile;
	/// mutex for eeg data file
	boost::mutex m_file_mutex;



public:
	Amplifier_File(AmplifierConfig &config);
	~Amplifier_File();
	EEGData* getImpedance() ;
	EEGData* getData() ;
	EEGHeader* getHeader();
	bool openDevice();
	bool closeDevice();
	bool beginAquisition();
	bool endAquisition();
};

#endif

