#ifndef EEGACCESS_H
#define EEGACCESS_H


#include "Amplifier.h"
#include "Amplifier_File.h"
#include "AmplifierConfig.h"
#include "EEGData.h"
//
#include <boost/bind.hpp>
#include <boost/utility.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>

class EEGAccess
{
private:
	static const int GET_DATA_TIMEOUT = 20;
	/// a pointer to the amplifier we are currently using
	Amplifier *m_Amplifier;
	/// a pointer to the eeg data buffer
	EEGData *m_buffer;
	/// is the aquire data thread running?
	bool m_aquiringData;
	/// thread to handle data aquisition from the amp
	boost::thread* m_aquireThread;
	/// mutex for the buffer
	boost::mutex m_buffer_mutex;

	boost::mutex mutex;
	/// buffer empty condition
	boost::condition m_buffer_empty;

	/**
		This method is run under a new process (m_aquireThread) when data aquisition is begun
		and while m_aquiringData is true
		It retrieves data from m_amplifier and places it in m_buffer
	*/
	void aquireData();

public:
	EEGAccess();
	~EEGAccess(){};
	void init(AmplifierConfig config);

	/**
		Begin data aquisition from m_amplifier. Calls aquireData() with a new thread
	*/
	void beginAquisition();

	/**
		Ends data aquisition and closes m_amplifier
	*/
	void endAquisition();

	/**
		get data from m_amplifier
		will begin data aquisition if required
		@return eeg data
	*/
	EEGData* getData();

	/**
		return the header object
		@return eeg header 
	*/
	EEGHeader* getHeader();


	/**
		not implemented
	*/
	void getImpedance();
	/**
		not implemented
	*/
	void setMarker();
};

#endif

