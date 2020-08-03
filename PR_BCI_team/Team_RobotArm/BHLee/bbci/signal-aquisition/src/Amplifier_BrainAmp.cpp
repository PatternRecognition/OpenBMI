

#include "Amplifier_BrainAmp.h"

#include <stdio.h>
#include <io.h>
#include <conio.h>
#include <vector>



using namespace std;
HANDLE	DeviceAmp = INVALID_HANDLE_VALUE;	// Amplifier device

Amplifier_BrainAmp::Amplifier_BrainAmp(AmplifierConfig &config){
	m_config = config;
	UsbDevice = false;	
	DriverVersion = 0;	
	m_deviceOpen = false;
	m_aquiring = false;
	//amplifiers = { None, None, None, None };	

}


EEGData* Amplifier_BrainAmp::getImpedance() {
	EEGData* none;
	return none;

}


// Find amplifiers
// Modified variables: amplifiers
// Return: number of successive amplifiers starting from the first position.
int Amplifier_BrainAmp::FindAmplifiers()
{
	USHORT amps[4];
	DWORD dwBytesReturned;
	DeviceIoControl(DeviceAmp, IOCTL_BA_AMPLIFIER_TYPE, NULL, 0, amps, sizeof(amps), 
			&dwBytesReturned, NULL);

	// count the number of amps attached
	int nAmps = 4;
	for (int i = 0; i < 4; i++)
	{
		amplifiers[i] = (AmpTypes)amps[i];
		if (amplifiers[i] == None && i < nAmps)
			nAmps = i;
	}
	return nAmps;
}



// Open BrainAmp device, first check for USB adapter and if that is not found, for PCI/ISA 
// Modified variables: UsbDevice, DriverVersion
// Return: true -> device found, false -> nothing found.
bool Amplifier_BrainAmp::openDevice(){

	DWORD dwFlags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH;
	// First try USB box
	DeviceAmp = CreateFile(DEVICE_USB, GENERIC_READ | GENERIC_WRITE, 0, NULL, 
								OPEN_EXISTING, dwFlags, NULL);
	if (DeviceAmp != INVALID_HANDLE_VALUE)
		UsbDevice = true;
	else
	{
		// USB box not found, try PCI host ad		UsbDevice = false;;
		DeviceAmp = CreateFile(DEVICE_PCI, GENERIC_READ | GENERIC_WRITE, 0, NULL, 
											OPEN_EXISTING, dwFlags, NULL);
	}
	
	// Retrieve driver version
	if (DeviceAmp == INVALID_HANDLE_VALUE)
		return false;
	
	DriverVersion = 0;
	DWORD dwBytesReturned;
	DeviceIoControl(DeviceAmp, IOCTL_BA_DRIVERVERSION, NULL, 0, &DriverVersion, sizeof(DriverVersion), 
		&dwBytesReturned, NULL);

	
	// initialise the setup details
	Setup.nHoldValue = 0x0;		// Value without trigger
	Setup.nPoints = m_config.getBufferSize();	
	Setup.nChannels = m_config.getNumChannels();
	for (int i = 0; i < Setup.nChannels; i++)
		Setup.nChannelList[i] = i;
	
	dwBytesReturned = 0;
	if (!DeviceIoControl(DeviceAmp, IOCTL_BA_SETUP, &Setup, sizeof(Setup), NULL, 0, &dwBytesReturned, NULL))
	printf("Setup failed, error code: %u\n", ::GetLastError());

	// Pulldown input resistors for trigger input, (active high)
	unsigned short pullup = 0;
	dwBytesReturned = 0;
	if (!DeviceIoControl(DeviceAmp, IOCTL_BA_DIGITALINPUT_PULL_UP, &pullup, sizeof(pullup), NULL, 0, &dwBytesReturned, NULL))
		printf("Can't set pull up/down resistors, error code: %u\n", ::GetLastError());

	// Make sure that enough amps exist, otherwise a long timeout will occur.	
	int numRequiredAmps = (int)ceil(m_config.getNumChannels() / 32.0);
	int numAmps = FindAmplifiers();
 	if (numAmps < numRequiredAmps) 
	{
		printf("Required Amplifiers: %d, Connected Amplifiers: %d\n", numRequiredAmps, numAmps);
		return false;
	}
	m_deviceOpen = true;
	return m_deviceOpen;
	
}



bool Amplifier_BrainAmp::closeDevice(){
	DWORD dwBytesReturned = 0;
	if (!DeviceIoControl(DeviceAmp, IOCTL_BA_STOP, NULL, 0, NULL, 0, &dwBytesReturned, NULL)){
		printf("Stop failed, error code: %u\n", ::GetLastError());
		return false;
	}

	return true;
}

EEGHeader* Amplifier_BrainAmp::getHeader(){
		
	return m_config.getHeader();
}



bool Amplifier_BrainAmp::beginAquisition(){
	// Start acquisition
	DWORD dwBytesReturned = 0;
	long acquisitionType = 1;
	m_aquiring = true;
	if (!DeviceIoControl(DeviceAmp, IOCTL_BA_START, &acquisitionType, sizeof(acquisitionType), NULL, 0, 
		&dwBytesReturned, NULL)){
		printf("Start failed, error code: %u\n", ::GetLastError());
		m_aquiring = false;
	}
	
	return m_aquiring;
	
}

bool Amplifier_BrainAmp::endAquisition(){
	return closeDevice();
	
}

EEGData* Amplifier_BrainAmp::getData(){
	
	// create new data buffer
	EEGData* data = new EEGData();
	data->includeMarkers(true);
	
	int numChannels = m_config.getNumChannels();
	int numSamples = m_config.getBufferSize();
	// set the number of channels on the data buffer
	data->setNumChannels(numChannels);
	// Data including marker channel
	vector<short> pnData((numChannels + 1) * numSamples);

	unsigned short nLastMarkerValue = 0;
	unsigned nDataOffset = 0;
	unsigned int nMarkerNumber = 0;

	// Check for error
	int nTemp = 0;
	DWORD dwBytesReturned = 0;

	while(!dwBytesReturned){

		if (!DeviceIoControl(DeviceAmp, IOCTL_BA_ERROR_STATE, NULL, 0, &nTemp, sizeof(nTemp), &dwBytesReturned, NULL))
		{
			printf("Acquisition Error, GetLastError(): %d\n", ::GetLastError());
			return data;
		}

		if (nTemp != 0)
		{
			printf("Acquisition Error %d\n", nTemp);
			return data;
		}

		// Get data
		int nTransferSize = (int)pnData.size() * sizeof(short);
		if (!ReadFile(DeviceAmp, &pnData[0], nTransferSize, &dwBytesReturned,NULL))
		{
			printf("Acquisition Error, GetLastError(): %d\n", ::GetLastError());
			return data;
		}
		// if nothing is returned, sleep and try again
		if (!dwBytesReturned)
		{
			Sleep(1);
			continue;
		}
	}

	// loop through the data and add the samples to the data buffer
	// for each sample
	for(int i = 0; i < numSamples; i++ )
	{
		vector<double> sample;
		// for each channel in the sample and the marker channel
		for(int j = 0; j < numChannels + 1; j++){
			short val = pnData[i * (numChannels + 1) + j];
			sample.push_back(val);
		}		
		// add sample to buffer
		data->addSample(sample);
	}

		
	
	return data;
}

