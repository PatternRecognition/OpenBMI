#ifndef AMPLIFIER_GTEC_H
#define AMPLIFIER_GTEC_H

#include "Amplifier.h"
#include <windows.h>


#include "gUSBamp.h"
#pragma comment(lib,"gUSBamp.lib")


	/** The data we need for each gTec controller */
typedef struct _ControllerData
{
  HANDLE hdev;          /** The handle to the controller */
  HANDLE dataEvent;     /** The event for the data access */
  OVERLAPPED ov;        /** The structure for the data access */
  BYTE *buffer;         /** The buffer for the data from the controller */
} ControllerData;



/**
	This class inherits from the Amplifier class
	It implements the Amplifier funcitonality for the GTEC gUSBAmp class of EEG amplifier

*/
class Amplifier_GTEC : public Amplifier
{

protected:


	
	ControllerData *controllers;
	static void printError();
	int Amplifier_GTEC::calcBufferLength(int time, int samplingRate);
	#define CONTROLLER_CHANNELS  16
	int m_numAmps;
	int m_ampBufferSize;
	int m_scanlineSize;
	

public:
	
	Amplifier_GTEC(AmplifierConfig &config);
	~Amplifier_GTEC();
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

