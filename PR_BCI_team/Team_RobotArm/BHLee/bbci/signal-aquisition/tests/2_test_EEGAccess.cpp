

#include "../src/EEGAccess.h"
#include "../src/AmplifierConfig.h"
#include <iostream>
#include <boost/cstdint.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace boost;


using namespace std;
using std::ostringstream; // stream insertion operators

#define TestGtec 0
#define TestBV 0
#define TestFile 1


BOOST_AUTO_TEST_CASE(TEST_EEG_ACCESS_File){
	if(TestFile == 1){
		
		ostringstream outputString;

		int numChannels = 64;
		// details to go into header
		vector<string> channels(numChannels);
		for(int i=0; i<(int)channels.size(); i++){
			outputString << "the channel number " << i;
			channels[i] = outputString.str();
			outputString.str( "" );
		}
		int sampleRate = 500;
		// create header

		EEGHeader hIn(channels, sampleRate);
		EEGData dataIn;
		dataIn.includeMarkers(false);

		int samples = 64;
		dataIn.setNumChannels(hIn.getNumChannels());

		for(int t=0; t<samples; t++){
			vector<double> sample;
			for(int i=0; i<dataIn.getNumChannels(); i++){
				sample.push_back(sin((double)t)+i);
			}
			dataIn.addSample(sample);
		}
		//hIn.print();
		//dataIn.print();
 		fstream file;
		file.open("eegAccessTest.eeg", ios::out | ios::binary);
		hIn.setFileFormat(FileFormat::STANDARD);
		hIn.setSampleRate(500);
		hIn.serialise(file);
		dataIn.serialise(file, &hIn);
		file.close();

		AmplifierConfig config;
		config.setInFileName("eegAccessTest");
		config.setOutFileName("eegAccessTestOut");
		config.setAmpType(AmpType::File);
		config.setBufferSize(samples);
		
		EEGAccess access;
		access.init(config);
		EEGHeader* hOut = access.getHeader();
		EEGData* dataOut = access.getData();

		//hOut->print();
		//dataOut->print();

		BOOST_REQUIRE(hIn.getNumChannels() == hOut->getNumChannels());
		BOOST_REQUIRE(hIn.getSampleRate() == hOut->getSampleRate());
		for (int i=0; i< hIn.getNumChannels(); i++){
			BOOST_REQUIRE(hIn.getChannelName(i) == hOut->getChannelName(i));
		}

		BOOST_REQUIRE(dataIn.getNumChannels() == dataOut->getNumChannels());
		BOOST_REQUIRE(dataIn.getNumChannels() == dataOut->getNumSamples());
		vector<double> sample = dataOut->getNextSample();
		int t =0;
		while((int)sample.size()>0){
			// sample.size() - 1 to account for the marker column
			for(int i=0; i<(int)sample.size() -1; i++){
				BOOST_CHECK_CLOSE((sin((double)t)+i), sample[i], 1e-3);
				
			}
			sample = dataOut->getNextSample();
			t++;
		}
		access.endAquisition();
	}
}



BOOST_AUTO_TEST_CASE(TEST_EEG_ACCESS_BV){
	if(TestBV == 1){
		ostringstream outputString;

		int numChannels = 10;
		// details to go into header
		vector<string> channels(numChannels);
		for(int i=0; i<(int)channels.size(); i++){
			outputString << "the channel number " << i;
			channels[i] = outputString.str();
			outputString.str( "" );
		}
		int sampleRate = 500;
		// create header

		EEGHeader hIn(channels, sampleRate);
		

		AmplifierConfig config;
		
		config.setOutFileName("eegAccessTestOutBV");
		config.setHeader(&hIn);
		config.setFileFormat(FileFormat::BRAIN_VISION);
		//config.setOutFileName("eegAccessTestOut");
		config.setAmpType(AmpType::BrainAmp);
		config.setBufferSize(sampleRate);
		
		EEGAccess access;
		access.init(config);
		EEGHeader* hOut = access.getHeader();
		EEGData* dataOut = access.getData();

		
		access.endAquisition();
	}
}




BOOST_AUTO_TEST_CASE(TEST_EEG_ACCESS_GTEC){
	
	if(TestGtec == 1){
		ostringstream outputString;

		int numChannels = 16;
		// details to go into header
		vector<string> channels(numChannels);
		for(int i=0; i<(int)channels.size(); i++){
			outputString << "the channel number " << i;
			channels[i] = outputString.str();
			outputString.str( "" );
		}
		int sampleRate = 128;
		// create header

		EEGHeader hIn(channels, sampleRate);
		

		AmplifierConfig config;
		
		config.setOutFileName("eegAccessTestOutGTEC");
		config.setHeader(&hIn);
		config.setFileFormat(FileFormat::BRAIN_VISION);
		//config.setOutFileName("eegAccessTestOut");
		config.setAmpType(AmpType::Gtec);
		config.setSampleRate(sampleRate);
		config.setBufferSize(32);
		
		EEGAccess access;
		access.init(config);
		EEGHeader* hOut = access.getHeader();
		for(int i = 0; i<4; i++){
			EEGData* dataOut = access.getData();
			dataOut->print();
		}
		
		access.endAquisition();
		
	}
}


