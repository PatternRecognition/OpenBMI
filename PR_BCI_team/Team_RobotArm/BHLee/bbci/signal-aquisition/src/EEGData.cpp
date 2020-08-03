
#include "EEGData.h"
#include "Utilities.cpp"
#include <boost/cstdint.hpp>
#include <iostream>
using namespace std;
using namespace boost;
EEGData::EEGData(){

	m_numChannels = 0;
}

EEGData::EEGData(int startPos){

	m_numChannels = 0;
	m_dataStartPos = startPos;
}


int EEGData::getStartPos(){
	return m_dataStartPos;
}

void EEGData::setStartPos(int startPos){
	m_dataStartPos = startPos;
}

bool EEGData::hasMarkers(){
	return m_hasMarkers;
}

void EEGData::includeMarkers(bool includeMarkers){
	m_hasMarkers = includeMarkers;
}

int EEGData::getNumSamples(){
	return m_samples.size();
}

int EEGData::getNumChannels(){
	return m_numChannels;
}

void EEGData::setNumChannels(int numChannels){
	m_numChannels = numChannels;
}

void EEGData::addSample(vector<double> sample){
	int markerOffset = 0;
	if(m_hasMarkers)
		markerOffset = 1;

	if((int)sample.size() == (m_numChannels + markerOffset)){
		m_samples.push_back(sample);

	}else{
		throw "Number of channels in sample does not match number of channels in EEGData object";
	}
}

void EEGData::addSamples(vector< vector<double> > samples){
	for(int i=0; i<(int)samples.size(); i++){
		addSample(samples[i]);
	}
}


vector<double> EEGData::getNextSample(){
	if((int)m_samples.size() > 0){
		vector<double> sample = m_samples.front();
		m_samples.erase(m_samples.begin());
		return sample;
	}
	vector<double> none;
	return none;
}


EEGMarker* EEGData::extractMarkers(){
	EEGMarker *mrk = new EEGMarker;
	if(m_hasMarkers)
	{
		for(int i = 0; i<m_samples.size(); i++)
		{			
			int dataPos = m_dataStartPos + i;
			int chanVal = (int)m_samples[i].back();
			m_samples[i].pop_back();
			mrk->addSample(chanVal, dataPos);
		}
	}

	return mrk;
}


void EEGData::addMarkers(EEGMarker *markers){
	m_hasMarkers = true;
	for(int i = 0; i<m_samples.size(); i++)
	{			
		int dataPos = m_dataStartPos + i;
		if(markers->peek().pos == dataPos)
		{
			int val = markers->getNextSampleAsInt();
			m_samples[i].push_back(val);
		}
		else
		{
			m_samples[i].push_back(0);
		}
		
	}


}

vector< vector<double> > EEGData::getSamples(int numSamples){
	vector< vector<double> > samples;
	vector<double> sample;

	int i = 0;
	while(i < numSamples || numSamples == 0 ){
		sample = getNextSample();
		if ((int)sample.size() > 0)
			samples.push_back(sample);
		else
			break;
		i++;
	}
	return samples;

}


void EEGData::serialise(iostream &stream, EEGHeader *header){

	int binaryDataFormat = header->getBinaryDataFormat();
	vector<double> sample = getNextSample();
	while((int)sample.size() > 0){
		for(int i=0; i<(int)sample.size(); i++){
			double value = sample[i];
			if (binaryDataFormat==BinaryDataFormat::INT_16) {
				int16_t channelVal = (int16_t)value;
				stream.write((char*)&channelVal , sizeof(int16_t));			
			} else if (binaryDataFormat==BinaryDataFormat::INT_32) {
				int32_t channelVal = (int32_t)value;
				stream.write((char*)&channelVal , sizeof(int32_t));			
			} else if (binaryDataFormat==BinaryDataFormat::FLOAT_) {
				float channelVal = (float)value;
				stream.write((char*)&channelVal , sizeof(float));			
			} else if (binaryDataFormat==BinaryDataFormat::DOUBLE_) {
				stream.write((char*)&value , sizeof(double));			
			}
			
		}
		sample = getNextSample();
	}
	stream.flush();


}

/*************************************************************
 *
 * reads a data block from the eeg-file we assume, that the
 * eeg-file has following properties:
 *  DataFormat = BINARY
 *  DataOrientation = MULTIPLEXED
 *  BinaryFormat = INT_16, INT_32 or FLOAT_32
 *
 * With swap we will determine if the endianes of this machine is different
 * to the endianes of the data.
 *
 *
 * If we have an eeg-file with multiplexed data layout, the data is
 * packet channel wise:
 *
 * |Value1:Chan1|Value1:Chan2| ... |Value1:ChanX|Value2:Chan1|Value2:Chan2| ...
 * |ValueY:Chan1| ... |ValueY:ChanX|EOF
 *
 *************************************************************/




void EEGData::deserialise(iostream &stream, EEGHeader *header, int maxRead=0)
{

	
    /* We need to check this code on 64-bit machines it might not work.  */
	
	//fixme no endian swap yet        
	int samplesRead = 0;
	m_numChannels = header->getNumChannels();
	int binaryDataFormat = header->getBinaryDataFormat();
	bool swap = header->getEndian();
	


	while(maxRead == 0 || samplesRead < maxRead){
		vector<double> sample;
		for(int i=0; i<m_numChannels; i++){			
			if(stream.eof()){
				break;
			} else {
				if (binaryDataFormat==BinaryDataFormat::INT_16) {
					int16_t channelVal;
					stream.read ((char*)&channelVal, sizeof(int16_t));
					sample.push_back((double)channelVal);
					//if (swap) swap16( (char*) &(((int16_t *)dataBuffer)[i]) );
					//sample.push_back((double) ((int16_t *)dataBuffer)[i]);
					
				} else if (binaryDataFormat==BinaryDataFormat::INT_32) {
					int32_t channelVal;
					stream.read ((char*)&channelVal, sizeof(int32_t));
					sample.push_back((double)channelVal);
					
				} else if (binaryDataFormat==BinaryDataFormat::FLOAT_) {
					float channelVal;
					stream.read ((char*)&channelVal, sizeof(float));
					sample.push_back((double)channelVal);
					
				} else if (binaryDataFormat==BinaryDataFormat::DOUBLE_) {
					double channelVal;
					stream.read ((char*)&channelVal, sizeof(double));
					sample.push_back(channelVal);
				}
			}
		}
		if(sample.size() == m_numChannels)
			m_samples.push_back(sample);
		else
			break;
		samplesRead++;
	}
    

}



EEGData* EEGData::copy(){
	EEGData *theCopy = new EEGData;
	theCopy->setNumChannels(m_numChannels);
	theCopy->includeMarkers(m_hasMarkers);
	theCopy->addSamples(m_samples);
	theCopy->setStartPos(m_dataStartPos);
	return theCopy;

}


void EEGData::print(){
	//boost::mutex::scoped_lock lk(m_io_mutex);
	std::cout << "\nEEG DATA: ";
	std::cout << "\nnum samples: ";
	std::cout << (int)m_samples.size();
	std::cout << "\nSamples--------------------------------: \n";

	for(int i=0;i<(int)m_samples.size(); i++){
		vector<double> sample = m_samples[i];
		for(int j=0; j<(int)sample.size(); j++){
			std::cout << sample[j] << ", ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}
