
#include "EEGMarker.h"
#include "Utilities.cpp"
#include <boost/cstdint.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
using namespace std;
using namespace boost;
EEGMarker::EEGMarker(){

	
}



string EEGMarker::padStimVal(int val){
	string stim;
	std::stringstream valString;
	valString << val;
	if(val < 10)
		stim = "  " + valString.str();
	else if (val < 100)
		stim = " " + valString.str();
	else 
		stim = valString.str();

	return stim;

}

int EEGMarker::getNumSamples(){
	return m_samples.size();
}


void EEGMarker::addSample(marker sample){
	m_samples.push_back(sample);
}


void EEGMarker::addSample(int val, int dataPos){
	
	EEGMarker::marker sample;
	if(val > 0){
		if(val < 255)
		{
			sample.type = "Stimulus";		
			sample.val = "S" + padStimVal(val);
		}
		else
		{
			sample.type = "Response";
			val = val - 255;
			sample.val = "R" + padStimVal(val);
		}

		sample.pos = dataPos;	
		m_samples.push_back(sample);
	}
}



void EEGMarker::addSamples(vector< marker > samples){
	for(int i=0; i<(int)samples.size(); i++){
		addSample(samples[i]);
	}
}



EEGMarker::marker EEGMarker::getNextSample(){
	if((int)m_samples.size() > 0){
		marker sample = m_samples.front();
		m_samples.erase(m_samples.begin());
		return sample;
	}
	marker none;
	return none;
}


int EEGMarker::getNextSampleAsInt(){
	if((int)m_samples.size() > 0){
		marker sample = m_samples.front();
		m_samples.erase(m_samples.begin());
		int val = atoi(sample.val.substr(1,3).c_str());
		if(sample.type == "Response")
			val += 255;
		return val;

	}
	
	return 0;
}


vector< EEGMarker::marker > EEGMarker::getSamples(int numSamples){
	vector< marker > samples;
	marker sample;

	int i = 0;

	while(i < numSamples|| (numSamples == 0 && m_samples.size() > 0) ){
		sample = getNextSample();
		samples.push_back(sample);
		i++;
	
	}
	return samples;

}

EEGMarker::marker EEGMarker::peek(){
	return (EEGMarker::marker)m_samples.front();

}


void EEGMarker::serialise(iostream &stream, EEGHeader *header){

	stream << "Brain Vision Data Exchange Marker File, Version 1.0" << "\n";
	stream << "[Common Infos]" << "\n";
	stream << "Codepage=UTF-8" << "\n";
	stream << "DataFile=" << header->getDataFileName() << BV_HDR_EXT << "\n";
	stream << "[Marker Infos]" << "\n";
	stream << "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>," << "\n";
	stream << "; <Size in data points>, <Channel number (0 = marker is related to all channels)>" << "\n";
	stream << "; Fields are delimited by commas, some fields might be omitted (empty)." << "\n";
	stream << "; Commas in type or description text are coded as \"\\1\"." << "\n";
	int markerNum = 1;
	int dataPosition = 1;
	
	while(m_samples.size() > 0){
		marker sample = getNextSample();
								
				stream << "Mk" << markerNum << "=" << sample.type << ",";
				stream << sample.val  << ",";
				stream << sample.pos << ",1,0" << "\n";
				markerNum += 1;		
	}	
	stream.flush();
}





void EEGMarker::deserialise(iostream &stream)
{	
	string line;	
	bool readingSamples = false;
	int chanNum = 0;
	int lastDataPoint = 0;

	while(getline(stream,line)){
		if(line[0]==';')
		{
			continue;
		}
		else if(readingSamples)
		{			
			vector<string> tokenList;
			split(tokenList, line, is_any_of(","), token_compress_on);						
			
			marker sample;
			sample.val = tokenList[1];
			sample.pos = atoi(tokenList[2].c_str());
			int typeIdx = tokenList[0].find("=") + 1;
			sample.type = tokenList[0].substr(typeIdx);
			m_samples.push_back(sample);		

		}
		else if(line.find("[Marker Infos]") != string::npos)
		{
			readingSamples = true;			
		}
		//cout << line << "\n";
	}  

}



EEGMarker* EEGMarker::copy(){
	EEGMarker *theCopy = new EEGMarker;
	theCopy->addSamples(m_samples);
	return theCopy;

}


void EEGMarker::print(){
	
	std::cout << "\nEEG Marker: ";
	std::cout << "\nnum samples: ";
	std::cout << (int)m_samples.size();
	std::cout << "\nSamples--------------------------------: \n";

	for(int i=0;i<(int)m_samples.size(); i++){
		marker sample = m_samples[i];
		std::cout << sample.pos << ", "<< sample.type << ", "<< sample.val;
		std::cout << "\n";
	}
	
	std::cout << "\n";
}
