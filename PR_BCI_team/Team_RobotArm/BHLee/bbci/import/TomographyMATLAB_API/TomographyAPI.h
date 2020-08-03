/* NiRX Medical Technolgies */
/* Tomography System SDK  */

#pragma pack(push)
#pragma pack(1)

/* API Errors */
 	#define NO_ERR 0 /* No error. */
	#define ERR_UNKNOWN 550000 /* An unknown error has occured. */
	#define ERR_NOT_INITIALIZED 550001 /* Tomography API has not been initialized. */
	#define ERR_APILOCK_TIMEOUT 550002 /* A timeout occured whilst waiting for command lock. */
	#define ERR_CMD_TIMEOUT 550003 /* A timeout occured whilst waiting for a command response. */
	#define ERR_NET_TIMEOUT 550004 /* A network timeout occured. */
	#define ERR_DATA_TIMEOUT 550005 /* A timeout occured whislst waiting for data packets. */
	#define ERR_DATA_OVERFLOW 550006 /* Data overflow has occured - the frame buffer is full. */

/* Status Flags */
	#define FLAGS_ACQUIRING 1
	#define FLAGS_OVERFLOW 2
	#define FLAGS_TRANSMITTING 4
	#define FLAGS_CONNECTED 8
	#define FLAGS_CLIENT_DATA_OVERFLOW 16

/* Custom Data Types */
	/* VersionStruct, used for returning the API's version number */
	typedef struct {
		unsigned short Major;
		unsigned short Minor;
		unsigned short Fix;
		unsigned short Build;
		} versionStruct;

#ifdef __cplusplus
extern "C" {
#endif

/*API Functions */	
int __cdecl tsdk_initialize(void);
int __cdecl tsdk_close(void);
int __cdecl tsdk_connect(char address[], unsigned short tcpPort, int timeout);
int __cdecl tsdk_disconnect(void);
int __cdecl tsdk_getStatus(unsigned int *statusFlags, double *sampleRate);
int __cdecl tsdk_getChannels(int *sources, int *detectors, int *wavelengths);
int __cdecl tsdk_getName(unsigned short nameType, int index, char pName[], int nameLength);
int __cdecl tsdk_start(int sources[], int detectors[], int wavelenths[], int numSources, int numDetectors, int numWavelengths, int bufferSize, int *elementsPerFrame);
int __cdecl tsdk_stop(void);
int __cdecl tsdk_getNFrames(int reqFrames, int timeout, int *frameCount, double timestamps[], char ti[], float data[], int *dataBufferSize);
int __cdecl tsdk_getFramesAvail(int *frameCount);

/* Support Functions */
void __cdecl tsdk_util_getErrorMsg(int errorCode, char errorMessage[], int messageLength);
void __cdecl tsdk_util_getAPIVersion(versionStruct *APIVersion);
void __cdecl tsdk_util_getTimeString(double timestamp, char timeString[], int len);

#ifdef __cplusplus
} // extern "C"
#endif

#pragma pack(pop)

