/**
* @mainpage iViewX API Documentation
*
* The iView X SDK provides an interface for communication between 
* your application and iView X. It uses UDP over Ethernet to provide 
* maximum speed and minimum latency for data transfer. By using 
* iView X SDK the programmer does not have to take care about Ethernet 
* communication specific programming. The iView X SDK provides a large 
* set of functions to control SMI eye tracker's and retrieve data online. 
* It supports a growing number of programming languages and environments, 
* e.g. C/C++, .Net, Matlab, Visual Basic, E-Prime.
*
* Important note: To be able to exchange data between iView X and another 
* application with the iView X SDK an Ethernet connection has to be 
* established. Please consult according documentation on how to establish 
* an Ethernet connection between different computers (e.g. the iView X 
* user manual). Even when using iView X and the application on the same 
* PC an Ethernet connection has to be established. Normally this happens 
* via the so called localhost, 127.0.0.1. Please adjust IP address and 
* port settings in iView X and your application accordingly. 
*
* @author SMI GmbH
*/ 

/**
* @file iViewXAPI.h
*
* @brief The file contains the prototype declaration for all supported 
* functions and data structs the customer needs to use 
**/ 

#define DLLExport __declspec(dllexport) 


#define RET_SUCCESS						1
#define RET_NO_VALID_DATA				2
#define RET_CALIBRATION_ABORTED			3
#define ERR_COULD_NOT_CONNECT			100
#define ERR_NOT_CONNECTED				101
#define ERR_NOT_CALIBRATED				102
#define ERR_NOT_VALIDATED				103
#define ERR_WRONG_DEVICE				111
#define ERR_WRONG_PARAMETER				112
#define ERR_WRONG_CALIBRATION_METHOD	113
#define ERR_CREATE_SOCKET				121
#define ERR_CONNECT_SOCKET				122
#define ERR_BIND_SOCKET					123
#define ERR_DELETE_SOCKET				124
#define ERR_NO_RESPONSE_FROM_IVIEWX		131
#define ERR_INVALID_IVIEWX_VERSION		132
#define ERR_WRONG_IVIEWX_VERSION		133
#define ERR_WRONG_SDK_DLL			133
#define ERR_ACCESS_TO_FILE				171
#define ERR_SOCKET_CONNECTION			181
#define ERR_EMPTY_DATA_BUFFER			191	
#define ERR_RECORDING_DATA_BUFFER		192	
#define ERR_FULL_DATA_BUFFER			193	
#define ERR_IVIEWX_IS_NOT_READY			194	
#define ERR_IVIEWX_NOT_FOUND			201	


/*
* With these defines it is possible to setup the logging status 
* for the function "iV_Log". With "iV_Log" it is possible to observe the 
* communication between a user’s application and iView X and/or function 
* calls. Log levels can be combined (e.g. LOG_BUG | LOG_IV_COMMAND | LOG_ETCOM). 
*/ 
#define LOG_LEVEL_BUG					1	
#define LOG_LEVEL_iV_FCT				2	
#define LOG_LEVEL_ALL_FCT				4	
#define LOG_LEVEL_IV_COMMAND			8	
#define LOG_LEVEL_RECV_IV_COMMAND		16	


/* 
* With ET_PARAM_ and function "iV_SetTrackingParameter" it is possible 
* to change iView X tracking parameters, for example pupil threshold and 
* corneal reflex thresholds, eye image contours, and other parameters.
* 
* Important note: This function can strongly affect tracking stability of 
* your iView X system. Only experienced users should use this function. 
*/
#define ET_PARAM_EYE_LEFT				0
#define ET_PARAM_EYE_RIGHT				1
#define ET_PARAM_PUPIL_THRESHOLD		0
#define ET_PARAM_REFLEX_THRESHOLD		1
#define ET_PARAM_SHOW_AOI				2
#define ET_PARAM_SHOW_CONTOUR			3
#define ET_PARAM_SHOW_PUPIL				4
#define ET_PARAM_SHOW_REFLEX			5
#define ET_PARAM_DYNAMIC_THRESHOLD		6
#define ET_PARAM_PUPIL_AREA				11
#define ET_PARAM_PUPIL_PERIMETER		12
#define ET_PARAM_PUPIL_DENSITY			13
#define ET_PARAM_REFLEX_PERIMETER		14
#define ET_PARAM_REFLEX_PUPIL_DISTANCE	15


/*
* The enumeration ETDevice can be used in connection with 
* "iV_GetSystemInfo" to get information about which type of device is 
* connected to iView X. It is part of the "SystemInfoStruct".
* (NONE = 0, RED = 1, HiSpeed = 2, MRI/MEG = 3, HED = 4, Custom = 5) 
*/
enum ETDevice {NONE, RED, HiSpeed, MRI, HED, Custom};


/**
* @struct SystemInfoStruct
*
* @brief This struct provides information about the eyetracking system in use. 
*
* The struct contains the following information:
* samplerate:			sample rate of eyetracking system in use
* iV_MajorVersion:		major version number of iView X in use
* iV_MinorVersion:		minor version number of iView X in use
* iV_Buildnumber:		build number of iView X in use
* API_MajorVersion:		major version number of iView X SDK in use
* API_MinorVersion:		minor version number of iView X SDK in use
* API_Buildnumber:		build number of iView X SDK in use
* iV_ETDevice:			type of eyetracking device
*
* To update information in "SystemInfoStruct" use function "iV_GetSystemInfo".
*/ 
struct SystemInfoStruct
{
	int samplerate;					
	int iV_MajorVersion;			
	int iV_MinorVersion;			
	int iV_Buildnumber;				
	int API_MajorVersion;			
	int API_MinorVersion;			
	int API_Buildnumber;			
	enum ETDevice iV_ETDevice;		
};


/**
* @struct CalibrationPointStruct
*
* @brief This struct provides information about the current calibration point. 
*
* The struct contains the following information:
* number:		number of calibration point that is currently active
* positionX:	horizontal position of calibration point that is currently active
* positionY:	vertical position of calibration point that is currently active
* 
* To update information in "CalibrationPointStruct" use function 
* "iV_GetCurrentCalibrationPoint" during a calibration or validation procedure. 
*/ 
struct CalibrationPointStruct
{
	int number;							
	int positionX;						
	int positionY;						
};


/**
* @struct EyeDataStruct
*
* @brief This struct provides information about eye data.
*
* The struct contains the following information:
* gazeX:			horizontal gaze position [pixel]
* gazeY:			vertical gaze position [pixel]
* diam:				pupil diameter [pixel, mm]
* eyePositionX:		horizontal eye position relative to camera
* eyePositionY: 	vertical eye position relative to camera
* eyePositionZ:		distance to camera 
*
* "EyeDataStruct" is part of "SampleStruct". To update information 
* in "SampleStruct" use function "iV_GetSample".
*/ 
struct EyeDataStruct
{
	double gazeX;					
	double gazeY;					
	double diam;					
	double eyePositionX;			
	double eyePositionY;			
	double eyePositionZ;			
};


/**
* @struct SampleStruct
*
* @brief This struct provides information about gaze data samples. 
*
* The struct contains the following information:
* timestamp:		timestamp of the last gaze data sample [microseconds]
* leftEye:			eye data left eye
* rightEye:			eye data right eye
* planeNumber:		plane number of gaze data sample
*
* The data describes the last gaze data sample that has been calculated. 
* It will be updated when a new gaze data sample has been calculated.
* To update information in "SampleStruct" use function "iV_GetSample".
*/ 
struct SampleStruct
{
	long long timestamp;				
	EyeDataStruct leftEye;				
	EyeDataStruct rightEye;				
	int	planeNumber;					
};


/**
* @struct SampleStruct32
*
* @brief This struct provides information about gaze data samples. 
*
* The struct contains the following information:
* timestamp:		timestamp of the last gaze data sample [microseconds]
* leftEye:			eye data left eye
* rightEye:			eye data right eye
* planeNumber:		plane number of gaze data sample
*
* The data describes the last gaze data sample that has been calculated. 
* It will be updated when a new gaze data sample has been calculated.
* To update information in "SampleStruct32" use function "iV_GetSample32".
*/ 
struct SampleStruct32
{
	double timestamp;					
	EyeDataStruct leftEye;				
	EyeDataStruct rightEye;				
	int	planeNumber;					
};


/**
* @struct EventStruct
*
* @brief This struct provides information about the last eye event that has 
* been calculated. 
*
* The struct contains the following information:
* eventType:	type of eye event, 'F' for fixation (at the moment only 
*				fixations are supported)
* eye:			related eye, 'l' for left eye, 'r' for right eye
* startTime: 	start time of the event in microseconds
* endTime: 		end time of the event in microseconds
* duration: 	duration of the event in microseconds
* positionX:	horizontal position of the fixation event [pixel]
* positionY:	vertical position of the fixation event [pixel]
*
* The data describes the last eye event that has been calculated. It will be 
* updated when a new event has been calculated.
* To update information in "EventStruct" use function "iV_GetEvent".
*/ 
struct EventStruct
{
	char eventType;						
	char eye;							
	long long startTime;				
	long long endTime;					
	long long duration;					
	double positionX;					
	double positionY;					
};


/**
* @struct EventStruct32
*
* @brief This struct provides information about the last eye event that has 
* been calculated. 
*
* The struct contains the following information:
* eventType:	type of eye event, 'F' for fixation (at the moment only 
*				fixations are supported)
* eye:			related eye, 'l' for left eye, 'r' for right eye
* startTime: 	start time of the event in microseconds
* endTime: 		end time of the event in microseconds
* duration: 	duration of the event in microseconds
* positionX:	horizontal position of the fixation event [pixel]
* positionY:	vertical position of the fixation event [pixel]
*
* The data describes the last eye event that has been calculated. It will be 
* updated when a new event has been calculated.
* To update information in "EventStruct32" use function "iV_GetEvent32".
*/ 
struct EventStruct32
{
	char eventType;						
	char eye;							
	double startTime;					
	double endTime;						
	double duration;					
	double positionX;					
	double positionY;					
};


/**
* @struct AccuracyStruct
*
* @brief This struct provides information about the last validation. 
*
* The struct contains the following information:
* deviationLX:	horizontal deviation target - gaze position for left eye [°]
* deviationLY:	vertical deviation target - gaze position for left eye [°]
* deviationRX:	horizontal deviation target - gaze position for right eye [°]
* deviationRY:	vertical deviation target - gaze position for right eye [°]
*
* To update information in "AccuracyStruct" use function "iV_GetAccuracy".
*/ 
struct AccuracyStruct
{
	double deviationLX;				
	double deviationLY;				
	double deviationRX;				
	double deviationRY;				
};


/**
* @struct CalibrationStruct
*
* @brief Use this struct to customize calibration behavior.
*
* The struct contains the following information:
* method:				Select Calibration Method (default: 5) 
* visualization:		Set Visualization Status [0: visualization by external 
*						stimulus program 1: visualization by SDK (default)]  
* displayDevice:		Set Display Device 
*						[0: primary device (default), 1: secondary device]
* speed:				Set Calibration/Validation Speed [0: slow (default), 1: fast] 
* autoAccept:			Set Calibration/Validation Point Acceptance 
*						[1: automatic (default) 0: manual] 
* foregroundBrightness:	Set Calibration/Validation Target Brightness 
*						[0..255] (default: 20) 
* backgroundBrightness:	Set Calibration/Validation Background Brightness 
*						[0..255] (default: 239) 
* targetShape:			Set Calibration/Validation Target Shape 
*						[IMAGE = 0, CIRCLE1 = 1 (default), CIRCLE2 = 2, CROSS = 3]
* targetSize:			Set Calibration/Validation Target Size 
*						(default: 10 pixels) 
* targetFilename: 		Select Custom Calibration/Validation Target
*
* To set calibration parameters with "CalibrationStruct" use function 
* "iV_SetupCalibration".
*/ 
struct CalibrationStruct
{
	int method;						
	int visualization;				
	int displayDevice;				
	int speed;						
	int autoAccept;					
	int foregroundBrightness;		
	int backgroundBrightness;		
	int targetShape;				
	int targetSize;					
	char targetFilename[256];		
};


/**
* @struct REDStandAloneModeStruct
*
* @brief Use this struct to customize RED operation mode.
*
* The struct contains the following information:
* stimX:				horizontal stimulus calibration size [mm] 
* stimY:				vertical stimulus calibration size [mm] 
* stimHeightOverFloor:	distance floor to stimulus screen [mm]
* redHeightOverFloor:	distance floor to RED [mm]
* redStimDist:			distance RED to stimulus screen [mm]
* redInclAngle:			RED inclination angle [°]
*
* Setup RED operation mode parameters with "REDStandAloneModeStruct" use function 
* "iV_SetupREDStandAloneMode".
*/ 
struct REDStandAloneModeStruct
{
	int stimX;
	int stimY;
	int stimHeightOverFloor;
	int redHeightOverFloor;
	int redStimDist;
	int redInclAngle;
};



typedef int (CALLBACK *pDLLSetCalibrationPoint)(struct CalibrationPointStruct calibrationPoint);
typedef int (CALLBACK *pDLLSetSample)(struct SampleStruct rawDataSample);
typedef int (CALLBACK *pDLLSetEvent)(struct EventStruct eventDataSample);




/**
* @brief	validates the customer license (only for OEM devices) 
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
**/ 
DLLExport int __stdcall iV_SetLicense(char* licenseKey);


/**
* @brief	run iView X application and connects automatically (only on same computer) 
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_COULD_NOT_CONNECT	- failed to establish connection 
* @return	ERR_IVIEWX_NOT_FOUND	- failed to start iViewX application 
**/ 
DLLExport int __stdcall iV_Start();


/**
* @brief	disconnects and closes iViewX 
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_DELETE_SOCKET		- failed to delete sockets 
**/ 
DLLExport int __stdcall iV_Quit();


/**
* @brief	sends a remote command to iView X. Please refer to the iView X
*			help file for further information about remote commands.
* 
* @param	ETMessage - iView X remote command
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
*			ERR_WRONG_PARAMETER	- parameter out of range
**/ 
DLLExport int __stdcall iV_SendCommand(char etMessage[256]);


/**
* @brief	establishes a UDP connection to iView X.
*			"iV_Connect" will not return until connection has been established. 
*			If no connection can be established it will return after three seconds.
* 
* @param	SendIPAddress	- IP address of iView X computer 
* @param	SendPort		- port being used by iView X SDK for sending data to 
*							iView X 
* @param	RecvIPAddress	- IP address of local computer 
* @param	ReceivePort		- port being used by iView X SDK for receiving data 
*							from iView X
*
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_WRONG_PARAMETER		- parameter out of range
* @return	ERR_COULD_NOT_CONNECT	- failed to establish connection
**/ 
DLLExport int __stdcall iV_Connect(char sendIPAddress[16], int sendPort, char recvIPAddress[16], int receivePort);


/**
* @brief	checks if connection to iView X is still established 
* 
* @return	RET_SUCCESS			- intended functionality has been fulfilled 
* @return	ERR_NOT_CONNECTED	- no connection established
**/ 
DLLExport int __stdcall iV_IsConnected();


/**
* @brief	disconnects from iView X
*			"iV_Disconnect" will not return until the connection 
*			has been disconnected.
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_DELETE_SOCKET		- failed to delete sockets 
**/ 
DLLExport int __stdcall iV_Disconnect();


/**
* @brief	updates "systemInfoData" with current system information 
*
* @param	SystemInfoStruct - see reference information for "SystemInfoStruct"
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	RET_NO_VALID_DATA	- no new data available
**/
DLLExport int __stdcall iV_GetSystemInfo(struct SystemInfoStruct *systemInfoData);


/**
* @brief	sets iView X tracking parameters
*
* @param	ET_PARAM_EYE		- select specific eye 
* @param	ET_PARAM			- select parameter that shall be set 
* @param	value				- new value for selected parameter
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_WRONG_PARAMETER	- parameter out of range
**/
DLLExport int __stdcall iV_SetTrackingParameter(int ET_PARAM_EYE, int ET_PARAM, int value);


/**
* @brief	starts gaze data recording and scene video recording (if connected 
*			eyetracking device is "HED")
*			"iV_StartRecording" does not return until gaze and scene video 
*			recording is started
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_WRONG_DEVICE	- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_StartRecording();


/**
* @brief 	stops gaze data recording and scene video recording (if connected 
*			eyetracking device is "HED")
*			"iV_StopRecording" does not return until gaze and scene video 
*			recording is stopped 
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_WRONG_DEVICE	- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_StopRecording();


/**
* @brief	pauses gaze data recording and scene video recording (if connected 
*			eyetracking device is "HED")
*			"iV_ContinueRecording" does not return until gaze and scene video 
*			recording is continued 
*
* @param	etMessage			- text message to be written to data file
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_WRONG_DEVICE	- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_ContinueRecording(char etMessage[256]);


/**
* @brief 	pauses gaze data recording and scene video recording (if connected 
*			eyetracking device is "HED")
*			"iV_PauseRecording" does not return until gaze and scene video 
*			recording is paused
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_WRONG_DEVICE	- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_PauseRecording();

/**
* @brief 	clears the data buffer and scene video buffer (if connected 
*			eyetracking device is "HED").
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_WRONG_DEVICE	- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_ClearRecordingBuffer();
	

/**
* @brief 	writes data buffer and scene video buffer (if connected eyetracking 
*			device is "HED") to file "filename"
*			"iV_SaveData" will not return until the data has been saved
*
* @param	Filename		- filename of data files being created 
*							(.idf: eyetracking data, .avi: scene video data) 
* @param	Description	- optional experiment description 
* @param	User			- optional name of test person
* @param	Overwrite		- 0: do not overwrite file “filename” if it already exists 
*							1: overwrite file “filename” if it already exists
*
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_NOT_CONNECTED		- no connection established 
* @return	ERR_NO_DATA_RECORDED	- if recording buffer is empty 
* @return	ERR_WRONG_PARAMETER		- parameter out of range
**/
DLLExport int __stdcall iV_SaveData(char filename[256], char description[64], char user[64], int overwrite);


/**
* @brief 	sets calibration parameters
*
* @param	calibrationData	- see reference information for "CalibrationStruct"
*
* @return	RET_SUCCESS						- intended functionality has been fulfilled
* @return	ERR_WRONG_PARAMETER				- parameter out of range
* @return	ERR_WRONG_DEVICE				- eye tracking device required for this 
*											function is not connected  
* @return	ERR_WRONG_CALIBRATION_METHOD	- eye tracking device required for this 
*											calibration method is not connected
**/
DLLExport int __stdcall iV_SetupCalibration(struct CalibrationStruct *calibrationData);


/**
* @brief 	stores a performed calibration 
*
* @param	name - calibration name / identifier 
*
* @return	RET_SUCCESS					- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED			- no connection established
* @return	ERR_NOT_CALIBRATED			- system is not calibrated
* @return	ERR_WRONG_IVIEWX_VERSION	- wrong version of iView X  
* @return	ERR_WRONG_DEVICE			- eye tracking device required for this 
*										function is not connected
**/
DLLExport int __stdcall iV_SaveCalibration(char name[256]);


/**
* @brief 	loads a saved calibration 
*			a calibration has to be previously saved by using "iV_SaveCalibration"
*
* @param	name - calibration name / identifier 
*
* @return	RET_SUCCESS					- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED			- no connection established
* @return	ERR_WRONG_IVIEWX_VERSION	- wrong version of iView X  
* @return	ERR_WRONG_DEVICE			- eye tracking device required for this 
*										function is not connected
* @return	ERR_NO_RESPONSE_FROM_IVIEWX - no response from iView X; check 
*										calibration name / identifier
**/
DLLExport int __stdcall iV_LoadCalibration(char name[256]);


/**
* @brief	accepts a calibration point (participat has to be tracked; 
*										only if calibration is active) 
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_NOT_CONNECTED		- no connection established
* @return	ERR_WRONG_DEVICE		- eye tracking device required for this 
*									function is not connected
**/ 
DLLExport int __stdcall iV_AcceptCalibrationPoint();


/**
* @brief	aborts a calibration or validation (only if calibration or validation is active) 
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_NOT_CONNECTED		- no connection established
* @return	ERR_WRONG_DEVICE		- eye tracking device required for this 
*									function is not connected
**/ 
DLLExport int __stdcall iV_AbortCalibration();


/**
* @brief	changes the position of a calibration point 
* 
* @param	number		- selected calibration point
* @param	positionX	- new X position on screen
* @param	positionY	- new Y position on screen
*
* @return	RET_SUCCESS					- intended functionality has been fulfilled 
* @return	ERR_NOT_CONNECTED			- no connection established
* @return	ERR_NO_RESPONSE_FROM_IVIEWX - no response from iView X; check 
*										calibration name / identifier
**/ 
DLLExport int __stdcall iV_ChangeCalibrationPoint(int number, int positionX, int positionY);


/**
* @brief	resets the default positions of all calibration points 
* 
* @return	RET_SUCCESS				- intended functionality has been fulfilled 
* @return	ERR_NOT_CONNECTED		- no connection established
**/ 
DLLExport int __stdcall iV_ResetCalibrationPoints();


/**
* @brief 	starts a calibration procedure.
*			If "CalibrationStruct::visualization" is set to "1" with "iV_SetupCalibration" 
*			"iV_Calibrate" will not return until the calibration has been finished or aborted. 
*
* @return	RET_SUCCESS					- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED			- no connection established
* @return	ERR_WRONG_DEVICE 			- eye tracking device required for this function 
*										is not connected
* @return	ERR_WRONG_CALIBRATION_METHOD - eye tracking device required for this 
*										calibration method is not connected
**/
DLLExport int __stdcall iV_Calibrate();


/**
* @brief 	starts a validation procedure.
*			If "CalibrationStruct::visualization" is set to "1" with "V_SetupCalibration" 
*			"iV_Calibrate" will not return until the calibration has been finished or aborted. 
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return ERR_NOT_CONNECTED		- no connection established
* @return ERR_NOT_CALIBRATED	- system is not calibrated
* @return ERR_WRONG_DEVICE		- eye tracking device required for this function is not connected
**/
DLLExport int __stdcall iV_Validate();


/**
* @brief 	updates "accuracyData" with current accuracy data 
*			If parameter "visualization" is set to "1" the accuracy data will be 
*			visualized in a dialog window
*			iV_GetAccuracy will not return until "AccuracyStruct" is updated 
*
* @param	accuracyData	- see reference information for "AccuracyStruct"
* @param	visualization	- 0: no visualization 
*							1: accuracy data will be visualized in a dialog window 
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
* @return	ERR_NOT_CALIBRATED	- system is not calibrated
* @return	ERR_WRONG_PARAMETER	- parameter out of range
* @return	ERR_NO_VALID_DATA	- no new data available; all values inside struct are set to -1
**/
DLLExport int __stdcall iV_GetAccuracy(struct AccuracyStruct *accuracyData, int visualization);


/**
* @brief 	updates "currentCalibrationPoint" with current calibration point data
*
* @param	actualCalibrationPoint	- see reference information for "CalibrationPointStruct"
*
* @return	RET_SUCCESS 		- intended functionality has been fulfilled
* @return	RET_NO_VALID_DATA	- no new data available
* @return	ERR_NOT_CONNECTED	- no connection established
**/
DLLExport int __stdcall iV_GetCurrentCalibrationPoint(struct CalibrationPointStruct *actualCalibrationPoint);


/**
* @brief 	enables a gaze data filter. This API bilateral filter was implemented
*			due to special HCI application requirements 
*
* @return	RET_SUCCESS		- intended functionality has been fulfilled
**/
DLLExport int __stdcall iV_EnableGazeDataFilter();


/**
* @brief 	disables the raw data filter
*
* @return	RET_SUCCESS		- intended functionality has been fulfilled
**/
DLLExport int __stdcall iV_DisableGazeDataFilter();


/**
* @brief 	updates "rawDataSample" with current eyetracking data
*
* @param	rawDataSample	- see reference information for "SampleStruct"
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	RET_NO_VALID_DATA	- no new data available
* @return	ERR_NOT_CONNECTED	- no connection established
**/
DLLExport int __stdcall iV_GetSample(struct SampleStruct *rawDataSample);


/**
* @brief 	updates "rawDataSample" with current eyetracking data
*
* @param	rawDataSample	- see reference information for "SampleStruct32"
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	RET_NO_VALID_DATA	- no new data available
* @return	ERR_NOT_CONNECTED	- no connection established
**/
DLLExport int __stdcall iV_GetSample32(struct SampleStruct32 *rawDataSample);


/**
* @brief 	defines detection parameter for online fixation detection algorithm
*
* @param	minDuration			- minimun fixation duration [ms]
*			maxDispersion		- maximum dispersion [px] for head tracking systems
*								or [deg] for non head tracking systems
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
**/
DLLExport int __stdcall iV_SetEventDetectionParameter(int minDuration, int maxDispersion);


/**
* @brief 	updates "eventDataSample" with current event data
*
* @param	eventDataSample		- see reference information for "EventStruct"
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	RET_NO_VALID_DATA	- no new data available
* @return	ERR_NOT_CONNECTED	- no connection established
**/
DLLExport int __stdcall iV_GetEvent(struct EventStruct *eventDataSample);


/**
* @brief 	updates "eventDataSample" with current event data
*
* @param	eventDataSample		- see reference information for "EventStruct32"
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	RET_NO_VALID_DATA	- no new data available
* @return	ERR_NOT_CONNECTED	- no connection established
**/
DLLExport int __stdcall iV_GetEvent32(struct EventStruct32 *eventDataSample);


/**
* @brief 	sends a text message to iView X. "etMessage" will be written 
*			to the data file. If "etMessage" ends on .jpg, .bmp, .png, or .avi 
*			BeGaze will separate the data buffer into according trials.
*
* @param	etMessage	- Any text message to separate trials (image name 
*			containing extensions) or any idf data marker 
*
* @return	RET_SUCCESS			- intended functionality has been fulfilled
* @return	ERR_NOT_CONNECTED	- no connection established
**/
DLLExport int __stdcall iV_SendImageMessage(char etMessage[256]);


/**
* @brief 	defines the logging behavior of iView X SDK
*
* @param	logLevel	- logging status, see "Explanations for Defines"
*						in this manual for further information
* @param	filename	- filename of log file
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return ERR_WRONG_PARAMETER	- parameter out of range
* @return ERR_ACCESS_TO_FILE	- failed to access log file
**/
DLLExport int __stdcall iV_SetLogger(int logLevel, char filename[256]);


/**
* @brief 	writes "logMessage" to log file 
*
* @param	logMessage	- message that shall be written to the log file 
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return ERR_ACCESS_TO_FILE	- failed to access log file
**/
DLLExport int __stdcall iV_Log(char logMessage[256]);


/**
* @brief 	defines remotely the RED stand alone mode 
*
* @param	standAloneModeGeometry	- see reference information for 
*									"REDStandAloneModeStruct"
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return ERR_NOT_CONNECTED		- no connection established
* @return ERR_WRONG_PARAMETER	- parameter out of range
* @return ERR_WRONG_DEVICE		- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_SetupREDStandAloneMode(struct REDStandAloneModeStruct *standAloneModeGeometry);


/**
* @brief 	requests the eye tracker time stamp
*
* @param	actualTimestamp		- provides the internal time stamp 
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return RET_NO_VALID_DATA		- no new data available
* @return ERR_NOT_CONNECTED		- no connection established
**/
DLLExport int __stdcall iV_GetActualTimestamp(long long *actualTimestamp);


/**
* @brief 	visualizes eye image in seperate dialog (available for all devices except RED)
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return ERR_NOT_CONNECTED		- no connection established
* @return ERR_WRONG_DEVICE		- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_ShowEyeImageMonitor();


/**
* @brief 	visualizes scene video in seperate dialog (available for HED devices only) 
*
* @return RET_SUCCESS			- intended functionality has been fulfilled
* @return ERR_NOT_CONNECTED		- no connection established
* @return ERR_WRONG_DEVICE		- eye tracking device required for this function 
*								is not connected
**/
DLLExport int __stdcall iV_ShowSceneVideoMonitor();


/**
* @brief 	visualizes RED Tracking Monitor (available for RED devices only)
*
* @return RET_SUCCESS		- intended functionality has been fulfilled
* @return ERR_NOT_CONNECTED	- no connection established
* @return ERR_WRONG_DEVICE	- eye tracking device required for this function 
*							is not connected
**/
DLLExport int __stdcall iV_ShowTrackingMonitor(); 


/**
* @brief 	"iV_CalibrationCallback" function will be called if a calibration 
*			point has changed, the calibration has been finished or aborted. 
*			This callback allows drawing a customized calibration routine. 
*
* @param	pCalibrationCallbackFunction - pointer to CalibrationCallbackFunction
**/
DLLExport void __stdcall iV_SetCalibrationCallback(pDLLSetCalibrationPoint pCalibrationCallbackFunction); 


/**
* @brief 	"iV_SampleCallback" function will be called if iView X has 
*			generated a new raw data sample. 
*			Important note: Dependent on the sample rate critical algorithms 
*			with high processor usage shouldn't be running within this callback 
*
* @param	pSampleCallbackFunction - pointer to SampleCallbackFunction
**/
DLLExport void __stdcall iV_SetSampleCallback(pDLLSetSample pSampleCallbackFunction); 


/**
* @brief 	"iV_EventCallback" function will be called if an real-time 
*			detected fixation has started or ended. 
*
* @param	pEventCallbackFunction - pointer to EventCallbackFunction
**/
DLLExport void __stdcall iV_SetEventCallback(pDLLSetEvent pEventCallbackFunction); 


/**
* @brief Test	- test routine 
**/
DLLExport int __stdcall Test(); 




