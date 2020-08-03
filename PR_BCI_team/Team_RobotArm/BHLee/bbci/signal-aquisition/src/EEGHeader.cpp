
#include "EEGHeader.h"
#include "Utilities.cpp"
#include <iostream>
#include <string>
using namespace std;

EEGHeader::EEGHeader(vector<string> channelNames, int sampleRate){
	int channelNum = (int)channelNames.size();
	m_channelNames.resize(channelNum);

	for(int i=0;i<channelNum; i++){
		m_channelNames[i] = channelNames[i];
	}
	m_sampleRate = sampleRate;
	init();
}

EEGHeader::EEGHeader(){

	m_sampleRate = 0;
	init();
}


void EEGHeader::init(){

	m_endian = getEndian();
	m_binaryDataFormat = BinaryDataFormat::DOUBLE_;
	m_FileFormat = FileFormat::STANDARD;

}


int EEGHeader::getNumChannels(){
	return (int)m_channelNames.size();
}


void EEGHeader::setChannelNames(vector<string> channelNames){
	int channelNum = (int)channelNames.size();
	m_channelNames.resize(channelNum);

	for(int i=0;i<channelNum; i++){
		m_channelNames[i] = channelNames[i];
	}
}

int EEGHeader::getSampleRate(){
	return m_sampleRate;
}

void EEGHeader::setSampleRate(int sampleRate){
	m_sampleRate = sampleRate;
}


string EEGHeader::getChannelName(int index){
	return m_channelNames[index];

}
void EEGHeader::serialise(iostream &stream){

	
	if(m_FileFormat == FileFormat::BRAIN_VISION)
		serialise_bv(stream);
	else if (m_FileFormat == FileFormat::STANDARD)
		serialise_standard(stream);

}

void EEGHeader::serialise_standard(iostream &stream){
	int headerSize = 0;
	stream.seekp(sizeof(int));

	// write binary data format
	stream.write((char*)(&m_binaryDataFormat) , sizeof(int));
	headerSize += sizeof(int);

	// write sample rate
	stream.write((char*)(&m_sampleRate) , sizeof(int));
	headerSize += sizeof(int);
    // write number channels
	int numChannels =(int) m_channelNames.size();
	stream.write((char*)(&numChannels) , sizeof(int));
	headerSize += sizeof(int);

	// write channels
	for(int i = 0; i<numChannels; i++){
		string channel = m_channelNames[i];
		int channelSize = channel.length()+1;//  ;sizeof(channel)

		stream.write((char*)(&channelSize) , sizeof(int));
		headerSize += sizeof(int);
		stream.write(channel.c_str()  , channelSize);
		headerSize += channelSize;
	
	}
	stream.seekp(0);
	stream.write((char*)(&headerSize) , sizeof(int));
	stream.seekp(headerSize + sizeof(int));
	stream.flush();
	
}


void EEGHeader::serialise_bv(iostream &stream){
	stream << "Brain Vision Data Exchange Header File Version 1.0" << "\n";
	stream << "; Data created by the EEGAccess BBCI" << "\n";
	stream << "[Common Infos]" << "\n";
	stream << "Codepage=UTF-8" << "\n";
	stream << "DataFile=" << m_DataFileName << BV_HDR_EXT << "\n";
	stream << "MarkerFile=" << m_DataFileName << BV_MRK_EXT << "\n";
	stream << "" << "\n";
	stream << "DataFormat=BINARY" << "\n";
	stream << "; Data orientation: MULTIPLEXED=ch1,pt1, ch2,pt1 ..." << "\n";
	stream << "DataOrientation=MULTIPLEXED" << "\n";
	stream << "NumberOfChannels=" << m_channelNames.size() << "\n";
	stream << "; Sampling interval in microseconds" << "\n";
	// convert from frequecy to sample interval in microseconds
	stream << "SamplingInterval=" << (1 / m_sampleRate * 1000000) << "\n"; 

	stream << "[Binary Infos]" << "\n";
	stream << "BinaryFormat=";
	if(m_binaryDataFormat == BinaryDataFormat::INT_16)
		stream << "INT_16" << "\n";
	else
		stream << "IEEE_FLOAT_32" << "\n";

	stream << "[Channel Infos]" << "\n";
	stream << "; Each entry: Ch<Channel number>=<Name>,<Reference channel name>," << "\n";
	stream << "; <Resolution in \"Unit\">,<Unit>, Future extensions.." << "\n";
	stream << "; Fields are delimited by commas, some fields might be omitted (empty)." << "\n";
	stream << "; Commas in channel names are coded as \"\\1\"." << "\n";

	// write channels
	for(int i = 0; i<m_channelNames.size(); i++){
		string channel = m_channelNames[i];
		stream << "Ch" << i+1 << channel << ",,,"  << "\n";
	
	}
	stream.flush();
	

}


void EEGHeader::deserialise(iostream &stream){

	if(m_FileFormat == FileFormat::BRAIN_VISION)
		deserialise_bv(stream);
	else if (m_FileFormat == FileFormat::STANDARD)
		deserialise_standard(stream);

}


void EEGHeader::deserialise_standard(iostream &stream){

	int headerSize;
	stream.read ((char*)(&headerSize), sizeof(int));

	// read binary data format
	stream.read ((char*)(&m_binaryDataFormat), sizeof(int));
	
	// read sample rate
	stream.read ((char*)(&m_sampleRate), sizeof(int));
	
	// read number channels
	int numChannels;
	
	stream.read ((char*)(&numChannels), sizeof(int));
	m_channelNames.clear();
	m_channelNames.resize(numChannels);


	// read channels
	for(int i = 0; i < m_channelNames.size(); i++){
		char *buff;
		int channelSize;
		stream.read ((char*)(&channelSize), sizeof(int));
		buff = new char[channelSize];
		stream.read(buff , channelSize);
		m_channelNames[i] = buff;
		
	}
}


void EEGHeader::deserialise_bv(iostream &stream){
	string line;
	bool readingChannels = false;
	int chanNum = 0;
	while(getline(stream,line)){
		if(line[0]==';'){
			continue;
		}else if(readingChannels){
			if(line.length()==0)
				readingChannels = false;
			else{
				int idxStart = line.find("=") +1;
				int len = line.find(",") - idxStart;
				string chanName = line.substr(idxStart, len);
				m_channelNames[chanNum] = chanName;
				chanNum += 1;
			}
		}else if(line.find("BinaryFormat") != string::npos){
			string format = line.substr(line.find("=") + 1);
			if(format == "INT_16")
				m_binaryDataFormat = BinaryDataFormat::INT_16;
			else if(format == "IEEE_FLOAT_32")
				m_binaryDataFormat = BinaryDataFormat::FLOAT_;				
		}else if(line.find("SamplingInterval") != string::npos){
			m_sampleRate = atoi(line.substr(line.find("=") + 1).c_str());			
		}else if(line.find("Channel Infos") != string::npos){
			readingChannels = true;
		}else if(line.find("NumberOfChannels") != string::npos){
			int numChannels = atoi(line.substr(line.find("=") + 1).c_str());			
			m_channelNames.clear();
			m_channelNames.resize(numChannels);
		}
		//cout << line << "\n";
	}
	
}


void EEGHeader::print(){

	std::cout << "\nnum channels: ";
	std::cout << (int)m_channelNames.size();
	std::cout << "\nSample Rate: ";
	std::cout << m_sampleRate;
	std::cout << "\nChannel Names: \n";
	for(int i=0;i<(int)m_channelNames.size(); i++){
		std::cout << m_channelNames[i] ;
		std::cout << "\n";

	}
	std::cout << "\n ";
	std::cout.flush();
}


char EEGHeader::getEndian(){

	return m_endian;
}

BinaryDataFormat EEGHeader::getBinaryDataFormat(){

	return m_binaryDataFormat;
}


void EEGHeader::setBinaryDataFormat(BinaryDataFormat format){

	m_binaryDataFormat= format;
}


FileFormat EEGHeader::getFileFormat(){

	return m_FileFormat;
}


void EEGHeader::setFileFormat(FileFormat format){

	m_FileFormat= format;
}


string EEGHeader::getDataFileName(){

	return m_DataFileName;
}


void EEGHeader::setDataFileName(string name){

	m_DataFileName= name;
}

