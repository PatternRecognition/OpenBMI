#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "TomographyAPI.h"

/*This test application connects to a server on a given address and port, displays all telemetry data avaliable, and then begins
printing published data to the console until either the connection is terminated by the server or it receives SIGINT (Ctrl-C).*/

typedef enum {error_parameters,error_api,error_memory} errorsT;

volatile int intFlag = 0;
void intHandler(int signal){
	/*Catch SIGINT*/
	printf("Shutting down.\r\n");
	intFlag = 1;
	return;
}

int main(int argc, char* argv[]){
	/*Counters*/
	int i;
	
	/*Connection params*/
	unsigned short port;
	int timeout;
	
	/*Streaming Params*/
	int sources;
	int detectors;
	int wavelengths;
	int frameSize;
	versionStruct version;
	
	/*Error handling*/
	errorsT errorCause;
	int errorCode;
	char* errorLocation = NULL;
	int initialized = 0;
	int connected = 0;
	int streaming = 0;
	
	/*Data*/
	int frameCount;
	int framesAvail;
	double* timestamps = NULL;
	char* timingBytes = NULL;
	float* data = NULL;
	int bufferSize;
	#define STRINGBUF_LEN 1024
	char* stringBuf = NULL;
	#define ERRORBUF_LEN 1024
	char* errorBuf = NULL;
	unsigned int statusFlags;
	double sampleRate;
	
	/*Make sure the argument count is correct.*/
	if(argc != 4){
		errorCause = error_parameters;
		goto error; /*Jump to error handler*/
	}
	
	/*Parse the port and timeout*/
	if(!sscanf(argv[2],"%hu",&port) || !sscanf(argv[3],"%d",&timeout)){
		/*Failed to parse*/
		errorCause = error_parameters;
		goto error; /*Jump to error handler*/
	}
	
	/*Set the interrupt handler*/
	signal(SIGINT,intHandler);
	
	/* Check API Version */
	tsdk_util_getAPIVersion(&version);
	printf("Version: %hu.%hu.%hu.%hu\r\n\r\n",version.Major,version.Minor,version.Fix,version.Build);
	
	/*Initialize the API*/
	errorCode = tsdk_initialize();
	if(errorCode){
		errorCause = error_api;
		errorLocation = "API Init";
		goto error; /*Jump to error handler*/
	}
	else{
		initialized = 1;
	}
	
	/*Connect*/
	errorCode = tsdk_connect(argv[1],port,timeout);
	if(errorCode){
		errorCause = error_api;
		errorLocation = "Connect";
		goto error; /*Jump to error handler*/
	}
	else{
		connected = 1;
	}
	
	/*Get the number of channels*/
	errorCode = tsdk_getChannels(&sources,&detectors,&wavelengths);
	if(errorCode){
		errorCause = error_api;
		errorLocation = "Get channel counts";
		goto error; /*Jump to error handler*/
	}
	
	/*Allocate the general string buffer*/
	stringBuf = (char*)malloc(STRINGBUF_LEN * sizeof(char));
	if(!stringBuf){
		errorCause = error_memory;
		goto error; /*Jump to error handler*/
	}
	
	/* Fetch Channel Names & Display to User */
	printf("Sources:\r\n");
	for(i=0;i<sources;i++){
		errorCode = tsdk_getName(0,i,stringBuf,STRINGBUF_LEN);
		if(errorCode){
			errorCause = error_api;
			errorLocation = "Names";
			goto error; /*Jump to error handler*/
		}
		printf("%s\r\n",stringBuf);
	}
	
	printf("\r\nDetectors:\r\n");
	for(i=0;i<detectors;i++){
		errorCode = tsdk_getName(1,i,stringBuf,STRINGBUF_LEN);
		if(errorCode){
			errorCause = error_api;
			errorLocation = "Names";
			goto error; /*Jump to error handler*/
		}
		printf("%s\r\n",stringBuf);
	}
	
	printf("\r\nWavelengths:\r\n");
	for(i=0;i<wavelengths;i++){
		errorCode = tsdk_getName(2,i,stringBuf,STRINGBUF_LEN);
		if(errorCode){
			errorCause = error_api;
			errorLocation = "Names";
			goto error; /*Jump to error handler*/
		}
		printf("%s\r\n",stringBuf);
	}	
	printf("\r\n");
	
	/*Start streaming (all channels)*/
	errorCode = tsdk_start(NULL,NULL,NULL,0,0,0,10,&frameSize);
	if(errorCode){
		errorCause = error_api;
		errorLocation = "Start";
		goto error; /*Jump to error handler*/
	}
	else{
		streaming = 1;
	}

	#define CHUNK_SIZE 1
	/*Allocate memory*/
	data = (float*)malloc(frameSize * sizeof(float));
	if(!data){
		errorCause = error_memory;
		goto error; /*Jump to error handler*/
	}
	bufferSize = frameSize;
	
	timestamps = (double*)malloc(CHUNK_SIZE * sizeof(double));
	if(!timestamps){
		errorCause = error_memory;
		goto error; /*Jump to error handler*/
	}
	
	timingBytes = (char*)malloc(CHUNK_SIZE * sizeof(char));
	if(!timingBytes){
		errorCause = error_memory;
		goto error; /*Jump to error handler*/
	}

	/*Begin trying to get data and dumping to the console*/
	while(!intFlag){ /*While not Ctrl-C*/
		bufferSize = frameSize;
		errorCode = tsdk_getNFrames(CHUNK_SIZE,timeout,&frameCount,timestamps,timingBytes,data,&bufferSize);
		if((errorCode != 0) && (errorCode != ERR_DATA_TIMEOUT)){
			errorCause = error_api;
			errorLocation = "getNFrames";
			goto error; /*Jump to error handler*/
		}
		if(frameCount){
			/*Get time string*/
			tsdk_util_getTimeString(timestamps[0],stringBuf,STRINGBUF_LEN);
			stringBuf[(STRINGBUF_LEN-1)] = 0; /*Ensure NULL termination.*/
			/*Get available frame count*/
			errorCode = tsdk_getFramesAvail(&framesAvail);
			if(errorCode){
				errorCause = error_api;
				errorLocation = "Frames Available";
				goto error; /*Jump to error handler*/
			}
			/*Output to console*/
			printf("--------\r\n[Frames remaining: %d]\r\n",framesAvail);
			printf("Time: %lf (%s)\r\n",timestamps[0],stringBuf);
			printf("Timing byte: 0x%02x\r\n",((unsigned int)(*(unsigned char*)&timingBytes[0])));
			printf("Data:\r\n");

			for(i=0;i<frameSize;i++)
			{
				printf("[%d]=%f\r\n",i,data[i]);
			}
			printf("--------\r\n\r\n");
		} 
		else{
			/*Timout.  Get status.*/
			errorCode = tsdk_getStatus(&statusFlags,&sampleRate);
			if(errorCode){
				errorCause = error_api;
				errorLocation = "Status";
				goto error; /*Jump to error handler*/
			}
			printf("Status Word: 0x%08x; Sample Rate: %lf\r\n",statusFlags,sampleRate);
			if(!(statusFlags & FLAGS_CONNECTED)){ /*Test connected*/
				printf("Connection lost.\r\n");
				intFlag=1;
			}
		}
	}
	
	/*Free memory*/
	free(timestamps);
	free(timingBytes);
	free(data);
	free(stringBuf);
	stringBuf = NULL;
	timestamps=NULL;
	timingBytes=NULL;
	data=NULL;
	
	/*Stop*/
	if(streaming){
		errorCode = tsdk_stop();
		if(errorCode){
			errorCause = error_api;
			errorLocation = "Stop";
			goto error; /*Jump to error handler*/
		}
		else{
			streaming = 0;
		}
	}
	
	/*Disconnect*/
	errorCode = tsdk_disconnect();
	if(errorCode){
		errorCause = error_api;
		errorLocation = "Disconnect";
		goto error; /*Jump to error handler*/
	}
	else{
		connected = 0;
	}
	
	/*Close the API*/
	errorCode = tsdk_close();
	if(errorCode){
		errorCause = error_api;
		errorLocation = "Close";
		goto error; /*Jump to error handler*/
	}
	
	/*Exit*/
	return(0);
	
	
	/*Error Handling block*/
	error: /*Errors come here for dispatch based on enum type.*/
	switch(errorCause){
		case error_parameters:
			printf("Usage: 'apitest {address} {port} {timeout}'");
			goto cleanup;
			
		case error_api:
			/*API Error*/
			errorBuf = (char*)malloc(ERRORBUF_LEN * sizeof(char));
			if(errorBuf){
				/*Memory OK*/
				tsdk_util_getErrorMsg(errorCode,errorBuf,ERRORBUF_LEN);
				errorBuf[(ERRORBUF_LEN-1)] = 0; /*Ensure NULL termination.*/
				printf("\r\n\r\n\r\nError Code: %d (%s) at '%s'\r\n",errorCode,errorBuf,errorLocation);
				free(errorBuf);
			}
			else{ /*Couldn't get memory*/
				printf("\r\n\r\n\r\nError Code: %d at '%s' and associated memory allocation falure.\r\n",errorCode,errorLocation);
			}
			goto cleanup;
			
		case error_memory:
			printf("Out of memory.\r\n");
			goto cleanup;
			
		default:
			goto cleanup;
	}
	
	cleanup: /*Generic cleanup actions.*/
	/*Close API.  Ignore the returns on API calls here.*/
	if(initialized){
		if(connected){
			if(streaming){
				tsdk_stop();
			}
			tsdk_disconnect();
		}
		tsdk_close();
	}
	/*Free memory*/
	if(timestamps){
		free(timestamps);
	}
	if(timingBytes){
		free(timingBytes);
	}
	if(data){
		free(data);
	}
	if(stringBuf){
		free(stringBuf);
	}
	/*Exit*/
	return(0);
}
