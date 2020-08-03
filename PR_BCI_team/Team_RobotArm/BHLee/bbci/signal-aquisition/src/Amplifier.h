#ifndef AMPLIFIER_H
#define AMPLIFIER_H

#include <string>
#include <vector>
#include "AmplifierConfig.h"
#include "EEGData.h"
#include "EEGHeader.h"



/**
This is a virtual class that is to be inherited by all amplifier types.
It defines the basic functionality that must be implemented by each amplifier.
*/
class Amplifier
{


protected:
	/// Holds the configuration details for the amplifier
	AmplifierConfig m_config;
	/// is the device open?
	bool m_deviceOpen;
	/// is the device currently aquiring data?
	bool m_aquiring;
	/// keeps track of the number of samples read
	int m_samplesRead;


public:
	/**
		Returns the buffer size being used by the amplifier. Buffer size defines the total
		number of data samples to be returned each time getData is called
		@return buffer size
	*/
	virtual int getBufferSize(){return m_config.getBufferSize();} ;
	~Amplifier() {};
	/**
		Read impedance levels from the amplifier
		@return impedance data
	*/
	virtual EEGData* getImpedance()=0;
	/**
		Return data from the amplifier
		@return eeg data
	*/
	virtual EEGData* getData()=0;
	/**
		Get the header used by this amplifier
		@return current amplifier header
	*/
	virtual EEGHeader* getHeader()=0;
	/**
		Open the amplifier device
		@return device opened successfully
	*/
	virtual bool openDevice()=0;
	/**
		Close the amplifier device
		@return device closed successfully
	*/
	virtual bool closeDevice()=0;
	/**
		Begin the aquisition process of the amplifier
		@return device aquisition began successfully
	*/
	virtual bool beginAquisition()=0;
	/**
		End the aquisition process of the amplifier
		@return device aquisition ended successfully
	*/
	virtual bool endAquisition()=0;
	/**
		Return the config object associated with this amp
		@return amplifier config
	*/
	virtual AmplifierConfig* getConfig(){ return &m_config;};
	/**
		Is the amplifier device open?
		@return amplifier decice is open
	*/
	virtual bool isOpen(){return m_deviceOpen;};
	/**
		Is the amplifier device aquiring data?
		@return amplifier device is aquiring data
	*/
	virtual bool isAquiring(){return m_aquiring;};


};

#endif

