#include "EEGAccess.h"
#include "Amplifier_File.h"
#include "Amplifier_BrainAmp.h"
#include "Amplifier_GTEC.h"
#include <iostream>
#include "Utilities.cpp"


EEGAccess::EEGAccess(){

	m_aquiringData = false;
}

void EEGAccess::init(AmplifierConfig config){

	// initialise the desired amplifier 
	AmpType ampType = config.getAmpType();
	if(ampType == AmpType::BrainAmp)
	{
		m_Amplifier = new Amplifier_BrainAmp(config);
	}
	else if(ampType == AmpType::File)
	{
		m_Amplifier = new Amplifier_File(config);
	}
	else if(ampType == AmpType::Gtec)
	{
		m_Amplifier = new Amplifier_GTEC(config);
	}

}

// for the thread
void EEGAccess::aquireData(){


	fstream dataFile;
	fstream markerFile;
	fstream headerFile;

	// only need serialise data to file if there is an outfile name
	bool writeOut = m_Amplifier->getConfig()->getOutFileName().length()>0;
	if(writeOut){

		// open the file to save header data to
		if(m_Amplifier->getConfig()->getFileFormat() == FileFormat::STANDARD)
		{
			dataFile.open((m_Amplifier->getConfig()->getOutFileName() + STD_EXT).c_str(), ios::out | ios::binary);
			m_Amplifier->getHeader()->serialise(dataFile);
			markerFile.open((m_Amplifier->getConfig()->getOutFileName() + STD_MRK_EXT).c_str(), ios::out);
		}
		else if(m_Amplifier->getConfig()->getFileFormat() == FileFormat::BRAIN_VISION)
		{
			headerFile.open((m_Amplifier->getConfig()->getOutFileName() + BV_HDR_EXT).c_str(), ios::out);
			// serialise the header to the beginning of the file
			m_Amplifier->getHeader()->serialise(headerFile);
			headerFile.close();
			dataFile.open((m_Amplifier->getConfig()->getOutFileName() + BV_EEG_EXT).c_str(), ios::out | ios::binary);
			markerFile.open((m_Amplifier->getConfig()->getOutFileName() + BV_MRK_EXT).c_str(), ios::out);
		}

		
		
	}

	EEGMarker *mrk = new EEGMarker();
	while(m_aquiringData)
	{		
		// get a lock on the buffer
		boost::mutex::scoped_lock lk(m_buffer_mutex);
		// get data from the amplifier
		EEGData *data = m_Amplifier->getData();
		EEGData *dataCopy;
		if(writeOut)
		{
			// make a copy of the data
			dataCopy = data->copy();
		}
		// add the data to the buffer
		m_buffer->addSamples(data->getSamples(0));
		// notifz that the buffer is not empty anymore
		m_buffer_empty.notify_all();

		if(writeOut)
		{
			if(dataCopy->hasMarkers())
			{	
				EEGMarker *temp = dataCopy->extractMarkers(); 
				mrk->addSamples(temp->getSamples(0));
			}
			// serialise the copy of the data to the data file
			dataCopy->serialise(dataFile, m_Amplifier->getHeader());
			delete dataCopy;
		}

		// cleanup		
		delete data;
		
	}
	mrk->serialise(markerFile, m_Amplifier->getHeader());
	delete mrk;

	if(dataFile.is_open())
		dataFile.close();
	if(headerFile.is_open())
		headerFile.close();
	if(markerFile.is_open())
		markerFile.close();
}

void EEGAccess::beginAquisition(){
	// check the amp status and open/begin aquisition if required
	if(!m_Amplifier->isOpen())
		m_Amplifier->openDevice();
	if(!m_Amplifier->isAquiring())
		m_Amplifier->beginAquisition();
	
	// create the buffer
	m_buffer = new EEGData();
	
	// set the number of channels
	m_buffer->setNumChannels(m_Amplifier->getHeader()->getNumChannels());
	m_aquiringData = true;
	// begin the aquisition thread
	m_aquireThread = new boost::thread(boost::bind(&EEGAccess::aquireData,this));
}

void EEGAccess::endAquisition(){
	// end the aquisition thread
	m_aquiringData = false;
	m_aquireThread->join();
	// cleanup and stop amplifier aquisition
	delete m_buffer;
	m_Amplifier->endAquisition();
	

	

}

EEGData* EEGAccess::getData(){
	// check that the amp is aquiring data
	if(!m_aquiringData)
		beginAquisition();

	int bufferSize = m_Amplifier->getConfig()->getBufferSize();
	int waitcount = 0;
	// get a lock on the buffer
	boost::mutex::scoped_lock lk(m_buffer_mutex);
	// if the buffer doesn|t have a full sample set yet, wait
	while(m_buffer->getNumSamples() < bufferSize){	
		// check if we've timed out. if so. get what data we can
		if (waitcount > GET_DATA_TIMEOUT)
			break;
		m_buffer_empty.wait(lk);
		waitcount++;
	}
	// object to hold the returned data
	EEGData *data = new EEGData();
	// setup and add samples
	data->setNumChannels(m_buffer->getNumChannels());
	data->addSamples(m_buffer->getSamples(bufferSize));
	
	return data;

}

EEGHeader* EEGAccess::getHeader(){	
	EEGHeader* h;
	h = m_Amplifier->getHeader();
	return h;
}

void EEGAccess::getImpedance(){


}

void EEGAccess::setMarker(){


}
