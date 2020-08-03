
#include "..\src\EEGHeader.h"
#include "..\src\EEGData.h"
#include "..\src\EEGMarker.h"
#include <iostream>
#include <boost/cstdint.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace boost;

//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE( Test_suite_EEGFileIO )



BOOST_AUTO_TEST_CASE(TEST_EEGHeader_Standard){
	int numChannels = 200;
	ostringstream outputString;
	vector<string> channels(numChannels);
	for(int i=0; i<(int)channels.size(); i++){
		outputString << "the channel number " << i;
		channels[i] = outputString.str();
		outputString.str( "" );
	}

	int sampleRate = 500;
	EEGHeader header;
	BOOST_REQUIRE(0 == header.getNumChannels());
	BOOST_REQUIRE(0 == header.getSampleRate());
	header.setSampleRate(sampleRate);
	BOOST_REQUIRE(sampleRate == header.getSampleRate());
	header.setChannelNames(channels);
	BOOST_REQUIRE(numChannels == header.getNumChannels());

	for(int i=0; i<numChannels; i++){
		outputString << "the channel number " << i;
		BOOST_REQUIRE(outputString.str().c_str(), header.getChannelName(i).c_str());
		outputString.str( "" );
	}

}

BOOST_AUTO_TEST_CASE(TEST_EEGHeader_IO){

	int sampleRate = 500;
	ostringstream outputString;

	// details to go into header
	vector<string> channels(100);
	for(int i=0; i<(int)channels.size(); i++){
		outputString << "the channel number " << i;
		channels[i] = outputString.str();
		outputString.str( "" );
	}
	// create header
	EEGHeader hIn(channels, sampleRate);

	


	for(int headForm = 1; headForm<2; headForm++){
		fstream file;
		if(headForm == 0){
			hIn.setFileFormat(FileFormat::BRAIN_VISION);			
			file.open("header.eeg", ios::out);
		}else if (headForm == 1){
			hIn.setFileFormat(FileFormat::STANDARD);
			file.open("header.eeg", ios::out | ios::binary);
		}
		// write header to file
		//hIn.print();

		
		hIn.serialise(file);
		file.close();

		EEGHeader hOut;
		if(headForm == 0){
			hOut.setFileFormat(FileFormat::BRAIN_VISION);
			file.open("header.eeg", ios::in);
		}else if (headForm == 1){
			hOut.setFileFormat(FileFormat::STANDARD);
			file.open("header.eeg", ios::in | ios::binary);
		}
		
		
		// read header from file
		hOut.deserialise(file);
		//hOut.print();
		file.close();


		// check equality
		BOOST_REQUIRE(hIn.getNumChannels() == hOut.getNumChannels());
		BOOST_REQUIRE(hIn.getSampleRate() == hOut.getSampleRate());
		for (int i=0; i< hIn.getNumChannels(); i++){
			BOOST_REQUIRE(hIn.getChannelName(i) == hOut.getChannelName(i));
		}


	}
}


BOOST_AUTO_TEST_CASE(TEST_EEGHeader_READ_BV){
	
	EEGHeader h;
	fstream file;	
	h.setFileFormat(FileFormat::BRAIN_VISION);			
	file.open("bv_header.vhdr", ios::in);	
	h.deserialise(file);
	file.close();	
	//h.print();
}


BOOST_AUTO_TEST_CASE(TEST_EEGHeader_WRITE_BV){
	
	int sampleRate = 500;
	ostringstream outputString;

	// details to go into header
	vector<string> channels(100);
	for(int i=0; i<(int)channels.size(); i++){
		outputString << "the channel number " << i;
		channels[i] = outputString.str();
		outputString.str( "" );
	}
	// create header
	EEGHeader hIn(channels, sampleRate);

	fstream file;	
	hIn.setFileFormat(FileFormat::BRAIN_VISION);			
	file.open("bv_headerOUT.vhdr", ios::out);	
	hIn.serialise(file);
	file.close();	

}


BOOST_AUTO_TEST_CASE(TEST_EEGHeader_IO_BV){
	
	EEGHeader h;
	fstream file;	
	fstream file2;	
	h.setFileFormat(FileFormat::BRAIN_VISION);			
	file.open("bv_header.vhdr", ios::in);	
	h.deserialise(file);
	file.close();	
	file2.open("bv_headerOUT221.vhdr", ios::out);	
	h.serialise(file);
	file2.close();	
}


BOOST_AUTO_TEST_CASE(TEST_EEGMarker_READ_BV){
	
	EEGMarker h;
	fstream file;		
	file.open("bv_marker.vmrk", ios::in);	
	h.deserialise(file);
	file.close();	
	//h.print();
}


BOOST_AUTO_TEST_CASE(TEST_EEGData_Standard){

	EEGData data;
	data.includeMarkers(false);
	int channels = 256;
	vector<double> sample;
	for(int i=0; i<channels; i++){
		sample.push_back(sin((double)2)+i);
	}

	BOOST_REQUIRE(0 == data.getNumChannels());
	BOOST_REQUIRE(0 == data.getNumSamples());
	try{
		data.addSample(sample);
		BOOST_FAIL( "Expected Exception" );  
	}catch(char *str){
		
	}
	
	BOOST_REQUIRE(0 == data.getNumSamples());
	data.setNumChannels(channels);
	BOOST_REQUIRE(channels == data.getNumChannels());
	try{
		data.addSample(sample);
	}catch(char *str){
		BOOST_FAIL( "Add sample failed" );  
	}
	
	BOOST_REQUIRE(1 == data.getNumSamples());
	vector<double> returned_sample = data.getNextSample();
	BOOST_REQUIRE(0 == data.getNumSamples());
	BOOST_REQUIRE(channels == data.getNumChannels());
	BOOST_REQUIRE((int)sample.size() == (int)returned_sample.size());
	for(int i=0; i<(int)sample.size(); i++){
			BOOST_CHECK_CLOSE(sample[i],returned_sample[i], 1e-3);
		}
}


BOOST_AUTO_TEST_CASE(TEST_EEGData_BV_IO){

	

	ostringstream outputString;

	// details to go into header
	vector<string> channels(64);
	for(int i=0; i<(int)channels.size(); i++){
		outputString << "the channel number " << i;
		channels[i] = outputString.str();
		outputString.str( "" );
	}

	int sampleRate = 500;

	// create header
	EEGHeader h(channels, sampleRate);
	h.setBinaryDataFormat(BinaryDataFormat::INT_16);

	int samples = 2;
	fstream file;
	file.open("chris_vis_count_1.eeg", ios::in | ios::binary);
	EEGData dataOut;
	dataOut.deserialise(file, &h, samples);
	//dataOut.print();
}

BOOST_AUTO_TEST_CASE(TEST_EEGData_IO){

	int num_channels = 256;

	// details to go into header
	ostringstream outputString;
	vector<string> channels(num_channels);
	for(int i=0; i<(int)channels.size(); i++){
		outputString << "the channel number " << i;
		channels[i] = outputString.str();
		outputString.str( "" );
	}

	int sampleRate = 500;

	// create header
	EEGHeader h(channels, sampleRate);

	for(int bin_form=0; bin_form<4; bin_form++){		

		if (bin_form==BinaryDataFormat::INT_16) {
			h.setBinaryDataFormat(BinaryDataFormat::INT_16);
		} else if (bin_form==BinaryDataFormat::INT_32) {
			h.setBinaryDataFormat(BinaryDataFormat::INT_32);
		} else if (bin_form==BinaryDataFormat::FLOAT_) {
			h.setBinaryDataFormat(BinaryDataFormat::FLOAT_);
		} else if (bin_form==BinaryDataFormat::DOUBLE_) {
			h.setBinaryDataFormat(BinaryDataFormat::DOUBLE_);
		}		
		
		// eegdata setup	
		EEGData dataIn;
		int samples = 100;
		dataIn.includeMarkers(false);
		dataIn.setNumChannels(num_channels);

		for(int t=0; t<samples; t++){
			vector<double> sample;
			for(int i=0; i<num_channels; i++){
				sample.push_back(sin((double)t)+i);
			}
			dataIn.addSample(sample);
		}

		// write header to file
		//dataIn.print();

		fstream file;
		file.open("data.eeg", ios::out | ios::binary);

		dataIn.serialise(file, &h);
		file.close();

		
		file.open("data.eeg", ios::in | ios::binary);
		EEGData dataOut;
		dataOut.deserialise(file, &h, samples);
		file.close();
		//dataOut.print();

		// check equality
		BOOST_REQUIRE(num_channels == dataOut.getNumChannels());
		BOOST_REQUIRE(samples == dataOut.getNumSamples());
		vector<double> sample = dataOut.getNextSample();
		int t =0;
		while((int)sample.size()>0){
			for(int i=0; i<(int)sample.size(); i++){
				if (bin_form==BinaryDataFormat::INT_16) 
				{
					BOOST_CHECK_CLOSE((int16_t)(sin((double)t)+i), sample[i], 1e-3);

				} else if (bin_form==BinaryDataFormat::INT_32) 
				{
					BOOST_CHECK_CLOSE((int32_t)(sin((double)t)+i), sample[i], 1e-3);

				} else if (bin_form==BinaryDataFormat::FLOAT_) 
				{
					BOOST_CHECK_CLOSE((sin((float)t)+i), sample[i], 1e-3);

				} else if (bin_form==BinaryDataFormat::DOUBLE_) 
				{
					BOOST_CHECK_CLOSE((sin((double)t)+i), sample[i], 1e-3);
				}
				
			}
			sample = dataOut.getNextSample();
			t++;
		}


	}

}



BOOST_AUTO_TEST_SUITE_END()

//
//TEST(EEGHeader_EEGData, IO) {
//	ostringstream outputString;
//
//	int numChannels = 2;
//	// details to go into header
//	vector<string> channels(numChannels);
//	for(int i=0; i<(int)channels.size(); i++){
//		outputString << "the channel number " << i;
//		channels[i] = outputString.str();
//		outputString.str( "" );
//	}
//
//	int sampleRate = 500;
//
//	// create header
//	EEGHeader hIn(channels, sampleRate);
//
//	// write header to file
//	//hIn.print();
//
//	fstream file;
//	file.open("header_data.eeg", ios::out | ios::binary);
//	hIn.serialise(file);
//
////	cout << "Written header!";
////	cout.flush();
////	EEGData dataIn;
////	int samples = 2;
////
////	dataIn.setNumChannels(hIn.getNumChannels());
////	cout << "will Data!";
////
////	cout.flush();
////	for(int t=0; t<samples; t++){
////		vector<double> sample;
////		for(int i=0; i<dataIn.getNumChannels(); i++){
////			sample.push_back(sin(t)+i);
////		}
////		dataIn.addSample(sample);
////	}
////
////	// write header to file
////	//dataIn.print();
////
////	dataIn.serialise(file);
//	file.close();
//
//	cout << "Written Data!";
//	cout.flush();
//	// read header from file
//	file.open("header_data.eeg", ios::in | ios::binary);
//
//	EEGHeader hOut;
//	hOut.deserialise(file);
//
//	// check equality
//	EXPECT_EQ(hIn.getNumChannels(), hOut.getNumChannels());
//	EXPECT_EQ(hIn.getSampleRate(), hOut.getSampleRate());
//	for (int i=0; i< hIn.getNumChannels(); i++){
//		EXPECT_EQ(hIn.getChannelName(i), hOut.getChannelName(i));
//	}
//
////	//hOut.print();
////	EEGData dataOut;
////	dataOut.deserialise(file, hIn.getNumChannels(), samples);
////	file.close();
////	//dataOut.print();
////
////
////	// check equality
////	EXPECT_EQ(numChannels, dataOut.getNumChannels());
////	EXPECT_EQ(samples, dataOut.getNumSamples());
////	vector<double> sample = dataOut.getNextSample();
////	int t =0;
////	while((int)sample.size()>0){
////		for(int i=0; i<(int)sample.size(); i++){
////			EXPECT_EQ((double)(sin(t)+i), sample[i]);
////		}
////		sample = dataOut.getNextSample();
////		t++;
////	}
//	cout << "DONE!";
//
//}
