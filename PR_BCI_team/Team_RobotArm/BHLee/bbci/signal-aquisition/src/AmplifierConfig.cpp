
#include "AmplifierConfig.h"
#include <iostream>

AmplifierConfig::AmplifierConfig(vector<string> channelNames, int sampleRate, int bufferSize, AmpType ampType, string inFile, string outFile){
	m_header = new EEGHeader(channelNames, sampleRate);
	m_inFile = inFile;
	m_outFile = outFile;
	m_ampType = ampType;
	m_bufferSize = bufferSize;

}

AmplifierConfig::AmplifierConfig(){

	m_header = new EEGHeader();
}

int AmplifierConfig::getNumChannels(){
	return m_header->getNumChannels();
}


void AmplifierConfig::setChannelNames(vector<string> channelNames){
	m_header->setChannelNames(channelNames);
}

int  AmplifierConfig::getBufferSize(){
	return m_bufferSize;
}
void  AmplifierConfig::setBufferSize(int bufferSize){
	m_bufferSize = bufferSize;
}

int AmplifierConfig::getSampleRate(){
	return m_header->getSampleRate();
}

void AmplifierConfig::setSampleRate(int sampleRate){
	m_header->setSampleRate(sampleRate);
}


string AmplifierConfig::getChannelName(int index){
	return m_header->getChannelName(index);

}

string  AmplifierConfig::getOutFileName(){
	return m_outFile;

}

void AmplifierConfig::setOutFileName(string outFile){
	m_outFile = outFile;
}
string  AmplifierConfig::getInFileName(){
	return m_inFile;
}

void AmplifierConfig::setInFileName(string inFile){
	m_inFile = inFile;
}

AmpType  AmplifierConfig::getAmpType(){
	return m_ampType;
}

void AmplifierConfig::setAmpType(AmpType ampType){
	m_ampType = ampType;
}

void  AmplifierConfig::setHeader(EEGHeader* header){
	m_header = header;
}

EEGHeader*  AmplifierConfig::getHeader(){
	return m_header;
}


FileFormat AmplifierConfig::getFileFormat(){
	return m_header->getFileFormat();
}


void AmplifierConfig::setFileFormat(FileFormat format){
	m_header->setFileFormat(format);
}


void AmplifierConfig::serialise(iostream &stream){

	// write the size of the infile member and then the member
	int inFileSize = sizeof(m_inFile);
	stream.write((char*)(&inFileSize) , sizeof(int));
	stream.write((char*)(&m_inFile) , inFileSize);

	// write the size of the outfile member and then the member
	int outFileSize = sizeof(m_outFile);
	stream.write((char*)(&outFileSize) , sizeof(int));
	stream.write((char*)(&m_outFile) , outFileSize);

	// write the amptype and buffersize
	stream.write((char*)(&m_ampType) , sizeof(int));
	stream.write((char*)(&m_bufferSize) , sizeof(int));

	// serialise the header
	m_header->serialise(stream);

}


void AmplifierConfig::deserialise(iostream &stream){
	// read the infile member
	int inFileSize;
	stream.read((char*)(&inFileSize) , sizeof(int));
	stream.read((char*)(&m_inFile) , inFileSize);

	// read the outfile member
	int outFileSize;
	stream.read((char*)(&outFileSize) , sizeof(int));
	stream.read((char*)(&m_outFile) , outFileSize);

	// read in amptype and buffersize
	stream.read((char*)(&m_ampType) , sizeof(int));
	stream.read((char*)(&m_bufferSize) , sizeof(int));

	// deserialise the header
	m_header->deserialise(stream);


}


void AmplifierConfig::print(){
	std::cout << "\nIn File: " << m_inFile;
	std::cout << "\nOut File: " << m_outFile;
	std::cout << "\nBuffer Size: " << m_bufferSize;
	m_header->print();

}

