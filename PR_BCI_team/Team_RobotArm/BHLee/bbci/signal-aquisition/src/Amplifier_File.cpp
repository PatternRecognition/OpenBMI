
#include "Amplifier_File.h"
#include "EEGData.h"
#include <fstream>
#include <iostream>



Amplifier_File::Amplifier_File(AmplifierConfig &config){
	m_config = config;
	m_deviceOpen = false;
	m_aquiring = false;
}


EEGData* Amplifier_File::getImpedance() {
	// not implemented yet
	EEGData* none;
	return none;

}
bool Amplifier_File::openDevice(){
	// if the file isn't open
	if(!m_dataFile.is_open()){
		try{
			// get the mutex
			boost::mutex::scoped_lock lk(m_file_mutex);
			fstream headerFile;
			fstream markerFile;

			// open the market file to read
			if(m_config.getFileFormat() == FileFormat::STANDARD)
				markerFile.open((m_config.getInFileName() + STD_MRK_EXT).c_str(), ios::in);
			else if(m_config.getFileFormat() == FileFormat::BRAIN_VISION)
				markerFile.open((m_config.getInFileName() + BV_MRK_EXT).c_str(), ios::in);

			m_markers.deserialise(markerFile);
			markerFile.close();

			// open the header file to read
			if(m_config.getFileFormat() == FileFormat::STANDARD)
			{
				m_dataFile.open((m_config.getInFileName() + STD_EXT).c_str(), ios::in | ios::binary);
				m_config.getHeader()->deserialise(m_dataFile);
			}
			else if(m_config.getFileFormat() == FileFormat::BRAIN_VISION)
			{
				headerFile.open((m_config.getInFileName() + BV_HDR_EXT).c_str(), ios::in);
				m_config.getHeader()->deserialise(headerFile);
				headerFile.close();
				m_dataFile.open((m_config.getInFileName() + BV_EEG_EXT).c_str(), ios::in | ios::binary);
			}
			
			
			m_samplesRead = 1;
			m_deviceOpen = true;
		}catch(exception ex){
			throw "Could Not Open File";
		}

	}
	return true;
	
}

bool Amplifier_File::closeDevice(){
	// close the file
	if(m_dataFile.is_open())
		m_dataFile.close();
	return true;
}

bool Amplifier_File::beginAquisition(){return true;}
bool Amplifier_File::endAquisition(){return true;}


EEGHeader* Amplifier_File::getHeader(){
	
	// open the file if needed
	if(!m_dataFile.is_open()){

		openDevice();
	}
	return m_config.getHeader();
}
EEGData* Amplifier_File::getData(){
	// open file if needed
	if(!m_deviceOpen){
		openDevice();
	}
	
	// create empty data buffer
	EEGData* data = new EEGData(m_samplesRead);
	// calculate pause time (the get data method should mimic real time data aquisition)
	int pause = (int)(m_config.getBufferSize() * 1/m_config.getSampleRate());
	// sleep
	boost::this_thread::sleep(boost::posix_time::milliseconds(pause));
	// get mutex and desearilise data
	boost::mutex::scoped_lock lk(m_file_mutex);
	data->deserialise(m_dataFile, m_config.getHeader(), m_config.getBufferSize());
	data->addMarkers(&m_markers);
	// increment the number of samples read
	m_samplesRead += m_config.getBufferSize();
	return data;
}

