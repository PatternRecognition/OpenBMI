#ifndef AMPLIFIER_BRAINAMP_H
#define AMPLIFIER_BRAINAMP_H

#include "Amplifier.h"
#include "BrainAmpIoCtl.h"
#include <windows.h>


#include <stdio.h>
#include <io.h>
#include <conio.h>
#include <vector>


/**
	This class inherits from the Amplifier class
	It implements the Amplifier funcitonality for the Brain Amp class of EEG amplifier

*/
class Amplifier_BrainAmp : public Amplifier
{

protected:
	/// Different amplifier types
	enum AmpTypes
	{
		None = 0, Standard = 1, MR = 2, DCMRplus = 3, ExG = 5
	};
	
	/// output filename
	string m_fileName;
	#define DEVICE_PCI		L"\\\\.\\BrainAmp"			// ISA/PCI device
	#define DEVICE_USB		L"\\\\.\\BrainAmpUSB1"		// USB device	
	/// Setup structure for brainamp amplifier
	BA_SETUP Setup;
	/// is usb device
	/// If true, the connected device is an USB box, otherwise a 
	/// PCI/ISA host adapter
	bool UsbDevice;			
	/// Driver version
	int	DriverVersion;		
	
	/// Connected amplifiers
	AmpTypes amplifiers[4];

	/**
		Find the number of connected amplifiers
		@return the number of connected brainamp amplifiers
	*/
	int FindAmplifiers();

public:
	Amplifier_BrainAmp(AmplifierConfig &config);
	~Amplifier_BrainAmp();	
	EEGData* getImpedance();
	EEGData* getData();
	EEGHeader* getHeader();
	AmplifierConfig* getConfig(){ return &m_config;};
	bool openDevice();
	bool closeDevice();
	bool beginAquisition();
	bool endAquisition();
};
#endif


