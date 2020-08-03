

#include "Amplifier_GTEC.h"
#include <fstream>
#include <iostream>

#include <math.h>

/** The includes for the gTec controllers */



Amplifier_GTEC::Amplifier_GTEC(AmplifierConfig &config){
	m_config = config;
	m_deviceOpen = false;	
	m_aquiring = false;

}


EEGData* Amplifier_GTEC::getImpedance() {
	EEGData* none;
	return none;

}

bool Amplifier_GTEC::openDevice(){
	/** Calculate the value for the buffersize with the default frequency */
  
  HANDLE hdev;
  int i, contrPos;
  DAC myDAC;
  
  
  /** set the basic values for every controller */  
  m_scanlineSize = (CONTROLLER_CHANNELS + 1) * sizeof(float);  /** this is only the size for one controller */
  m_ampBufferSize = m_config.getBufferSize() * m_scanlineSize;
  

  /** find the divices connected to the pc */
  m_numAmps = 0;
  for(i = 1; i < 16; ++i) {
    hdev = GT_OpenDevice(i);    
    if(NULL != hdev) {
      // found a divice
      ++m_numAmps;
      printf("Found a device at the position %i.\n",i);
      GT_CloseDevice(&hdev);
    } 
  }

  


  // check if we have enough amplifiers for the number of channels we want
  int numRequiredAmps = (int)(ceil(((float)m_config.getNumChannels() / CONTROLLER_CHANNELS)));  
  if (m_numAmps < numRequiredAmps){
	  printf("Required Amplifiers: %d, Connected Amplifiers: %d\n", numRequiredAmps, m_numAmps);
	  return false;
  }


  /** create the array for the controllers and initialize them*/
  controllers = (ControllerData *)malloc(sizeof(ControllerData) * m_numAmps);
  contrPos = 0;
  for(i = 1; i < 16; ++i) {
    hdev = GT_OpenDevice(i);
    
    if(NULL != hdev) {
      /** found a device */
      controllers[contrPos].hdev = hdev;
      controllers[contrPos].dataEvent = CreateEvent(NULL,FALSE,FALSE,NULL);
      controllers[contrPos].ov.hEvent = controllers[contrPos].dataEvent;
      controllers[contrPos].ov.Offset = 0;
      controllers[contrPos].ov.OffsetHigh = 0;
      controllers[contrPos].buffer = (BYTE*)malloc(HEADER_SIZE + m_ampBufferSize);
      
      /** init the device */

      if(FALSE == GT_SetSlave(hdev, 0 != contrPos)) { /** we assume that the first controller is the master */
        printError();
        return false;
      }
      if(FALSE == GT_SetSampleRate(hdev, (int)m_config.getSampleRate())) {
        printError();
        return false;
      }
	  if(FALSE == GT_SetBufferSize(hdev,m_config.getSampleRate())) {
        printError();
        return false;
      }
      if(FALSE == GT_EnableSC(hdev,TRUE)) {
        printError();
        return false;
      }

      if(FALSE == GT_SetMode(hdev,M_NORMAL)) {
         printError();
         return false;
      }
      
      /** uncomment for debugging
       * produces sine waves
       */ 
	  
      myDAC.Offset = 2047;
      myDAC.WaveShape = WS_SINE;
      myDAC.Frequency = 2;
      myDAC.Amplitude = 100;
      GT_SetDAC(hdev,myDAC);
      if(FALSE == GT_SetMode(hdev,M_CALIBRATE)) {
        printError();
        return false;
      }
      
      if(FALSE == GT_Start(hdev)) {
        printError();
        return false;
      }
      
      ++contrPos;
    } 
  }
  m_deviceOpen = true;
  return m_deviceOpen;
}



bool Amplifier_GTEC::closeDevice(){
	int curContrPos;

  
  /** delete the structures */
  for(curContrPos = m_numAmps - 1; curContrPos >= 0; --curContrPos) {
    ControllerData curContr = controllers[curContrPos];
    
    GT_Stop(curContr.hdev);
    CloseHandle(curContr.dataEvent);
    free(curContr.buffer);
  }
  
  free(controllers);
  m_numAmps = 0;
  return true;
}


bool Amplifier_GTEC::beginAquisition(){return openDevice();}
bool Amplifier_GTEC::endAquisition(){return closeDevice();}


EEGHeader* Amplifier_GTEC::getHeader(){
		
	return m_config.getHeader();
}


EEGData* Amplifier_GTEC::getData(){
	// create new data buffer
	EEGData* data = new EEGData();
	data->includeMarkers(true);
	/** Filter the data and create the output values */
	if(m_deviceOpen){
    
  
  
   
	  /** The values for the loop */
	  DWORD dwBytesReceived;              
	  int curContrPos; 
	  DWORD dwOVret;    
	  int numChannels = m_config.getNumChannels();
	  int numSamples = m_config.getBufferSize();
	  int overlappedVal;
	  //int iBytesperScan = pthis->m_numOfChannels*sizeof(float);

	  
	  
		/** send the data request for each controller */
		for(curContrPos = m_numAmps - 1; curContrPos >= 0; --curContrPos) {
		  /** set the overlapped structure for the data access*/
		  ResetEvent(controllers[curContrPos].dataEvent);
		  controllers[curContrPos].ov.hEvent = controllers[curContrPos].dataEvent;
		  controllers[curContrPos].ov.Offset = 0;
		  controllers[curContrPos].ov.OffsetHigh = 0;

		  /** Send the data request*/
		  if(FALSE == GT_GetData(controllers[curContrPos].hdev, controllers[curContrPos].buffer,HEADER_SIZE + m_ampBufferSize,&controllers[curContrPos].ov)) {
	        
			printError();
			return data;
		  }
		}
		
		int numberOfScanlines;
		/** wait for the data of each controller */
		for(curContrPos = m_numAmps - 1; curContrPos >= 0; --curContrPos) {
		  

		  /** wait for the data*/
		  dwOVret = WaitForSingleObject(controllers[curContrPos].dataEvent,1000);
		  if(dwOVret == WAIT_TIMEOUT)
		  {
			/** we have a timeout. This occours in most case when the sync cable is not properly
			 * connected */
			GT_ResetTransfer(controllers[curContrPos].hdev);
			//Beep(1200,100);
		  }

		  /** get the number of bytes written to the buffer */
		  dwBytesReceived = 0;
		  // TODO don't know if this is right.. but the return value shoudl be one. other values mean bad read
		  overlappedVal = GetOverlappedResult(controllers[curContrPos].hdev,&controllers[curContrPos].ov,&dwBytesReceived,FALSE);
		  if (1 != overlappedVal){
			  //Beep(1200,100);
			  return data;

		  }
		
			
		  dwBytesReceived -= HEADER_SIZE; 
		  if(curContrPos == 0)
			numberOfScanlines = (int)dwBytesReceived/(m_scanlineSize);
	      
		}
		
		 
		
		
		// set the number of channels on the data buffer
		data->setNumChannels(numChannels);
		
		// get the data from each scan in order
		for(int scanNum =0; scanNum<numberOfScanlines; scanNum++)
		{
			vector<double> sample;
			int channelCounter = 0;
			// go through each controller
			for(curContrPos = m_numAmps - 1; curContrPos >= 0; --curContrPos) {	
				float* data = (float *)(controllers[curContrPos].buffer + HEADER_SIZE);
				// go through each channel
				float *pfl = NULL;
				
				for(int chan =0; chan < CONTROLLER_CHANNELS + 1; chan++){
					
						pfl = (float*)(controllers[curContrPos].buffer + HEADER_SIZE + (m_scanlineSize * scanNum) + (chan * 4));
						double tempy = (double)*pfl;
						sample.push_back(tempy);	
						//TODO - markers.. 1 per amp?
				
					
				}
			}
			data->addSample(sample);
		}


	  
		
    
	}
	/*boost::xtime xt;
	boost::xtime_get(&xt, boost::TIME_UTC);
	xt.sec += 1;
	boost::thread::sleep(xt);*/
	Beep(440,100);
	
	return data;
}




/**************************************************************
 *
 * Prints the error text after a failed command for the gtec controller.
 *
 *************************************************************/
void Amplifier_GTEC::printError() {
	WORD errorCode;
	char *errorText = NULL;
	GT_GetLastError(&errorCode,errorText);
	printf("The error was: Error(%i): %s.\n",errorCode,errorText);
}
	

/**********************************************
 *
 * This funciton calculates the size of the buffer for given timeperiod and
 * sampling rate.
 * @param time The time in ms the buffer can hold
 * @param samplingRate The sampling rate of the gtec controller
 *
 * @return The size of the buffer for one gtec controller in scanlines
 *
 *********************************************/
int Amplifier_GTEC::calcBufferLength(int time, int samplingRate) {
  double size;
  
  size = (double)time * (double)samplingRate * 0.001;
  size = ceil(size);
  /** The controller cannot hold more than 512 scans */
  if(size > 512) {
    size = 512;
  }
  
  return (int)size;
}

