#pragma once

#include <vector>
#include "KinovaTypes.h"
#include "CommunicationLayerWindows.h"
#include <stdio.h>
#include <Windows.h>

//This defines the the location of the communication layer.(CommunicationLayerWindows.dll)
#define COMM_LAYER_PATH L"CommunicationLayerWindows.dll"
#define COMM_LAYER_ETHERNET_PATH L"CommunicationLayerEthernet.dll"


// ***** E R R O R   C O D E S ******
#define ERROR_INIT_API 2001      // Error while initializing the API
#define ERROR_LOAD_COMM_DLL 2002 // Error while loading the communication layer

//Those 3 codes are mostly for internal use
#define JACO_NACK_FIRST 2003
#define JACO_COMM_FAILED 2004
#define JACO_NACK_NORMAL 2005

//Unable to initialize the communication layer.
#define ERROR_INIT_COMM_METHOD  2006

//Unable to load the Close() function from the communication layer.
#define ERROR_CLOSE_METHOD  2007

//Unable to load the GetDeviceCount() function from the communication layer.
#define ERROR_GET_DEVICE_COUNT_METHOD  2008

//Unable to load the SendPacket() function from the communication layer.
#define ERROR_SEND_PACKET_METHOD  2009

//Unable to load the SetActiveDevice() function from the communication layer.
#define ERROR_SET_ACTIVE_DEVICE_METHOD 2010

//Unable to load the GetDeviceList() function from the communication layer.
#define ERROR_GET_DEVICES_LIST_METHOD 2011

//Unable to initialized the system semaphore.
#define ERROR_SEMAPHORE_FAILED 2012

//Unable to load the ScanForNewDevice() function from the communication layer.
#define ERROR_SCAN_FOR_NEW_DEVICE 2013

//Unable to load the GetActiveDevice function from the communication layer.
#define ERROR_GET_ACTIVE_DEVICE_METHOD 2014

//Unable to load the OpenRS485_Activate() function from the communication layer.
#define ERROR_OPEN_RS485_ACTIVATE 2015

//A function's parameter is not valid.
#define ERROR_INVALID_PARAM 2100

//The API is not initialized.
#define ERROR_API_NOT_INITIALIZED 2101

//Unable to load the InitDataStructure() function from the communication layer.
#define ERROR_INIT_DATA_STRUCTURES_METHOD 2102

// ***** E N D  O F  E R R O R   C O D E S ******


//This represents the size of an array containing Cartesian values.
#define CARTESIAN_SIZE 6

//This represents the max actuator count in our context.
#define MAX_ACTUATORS 7

//This represents the max actuator count in our context.
#define MAX_INVENTORY 15

//This represents the size of the array returned by the function GetCodeVersion.
#define CODE_VERSION_COUNT 42

//This represents the size of the array returned by the function GetAPIVersion.
#define API_VERSION_COUNT 3

//This represents the size of the array returned by the function GetPositionCurrentActuators.
#define POSITION_CURRENT_COUNT 12

#define POSITION_CURRENT_COUNT_7DOF 14

//This represents the size of the array returned by the function GetSpasmFilterValues and sent to SetSpasmFilterValues.
#define SPASM_FILTER_COUNT 1

//Version of the API 5.01.04
#define COMMAND_LAYER_VERSION 50200

#define COMMAND_SIZE 70

#define OPTIMAL_Z_PARAM_SIZE 16

#define OPTIMAL_Z_PARAM_SIZE_7DOF 19

#define GRAVITY_VECTOR_SIZE 3

#define GRAVITY_PARAM_SIZE 42

#define GRAVITY_PAYLOAD_SIZE 4

//This represents the size of the buffer for the IP address.
#define IP_ADDRESS_LENGTH 4
#define MAC_ADDRESS_LENGTH 6

// ***** API'S FUNCTIONAL CORE *****

extern "C" __declspec(dllexport) int GetDevices(KinovaDevice devices[MAX_KINOVA_DEVICE], int &result);

extern "C" __declspec(dllexport) int SetActiveDevice(KinovaDevice device);

extern "C" __declspec(dllexport) int SetActiveDeviceEthernet(KinovaDevice device, unsigned long ipAddress);

extern "C" __declspec(dllexport) int RefresDevicesList();

extern "C" __declspec(dllexport) int InitAPI(void);

extern "C" __declspec(dllexport) int InitEthernetAPI(EthernetCommConfig & config);

extern "C" __declspec(dllexport) int CloseAPI(void);

extern "C" __declspec(dllexport) int GetCodeVersion(int Response[CODE_VERSION_COUNT]);

extern "C" __declspec(dllexport) int GetAPIVersion(int Response[API_VERSION_COUNT]);

extern "C" __declspec(dllexport) int GetCartesianPosition(CartesianPosition &Response);

extern "C" __declspec(dllexport) int GetAngularPosition(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetCartesianForce(CartesianPosition &Response);

extern "C" __declspec(dllexport) int GetAngularForce(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetAngularCurrent(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetActualTrajectoryInfo(TrajectoryPoint &Response);

extern "C" __declspec(dllexport) int GetGlobalTrajectoryInfo(TrajectoryFIFO &Response);

extern "C" __declspec(dllexport) int GetSensorsInfo(SensorsInfo &Response);

extern "C" __declspec(dllexport) int GetSingularityVector(SingularityVector &Response);

extern "C" __declspec(dllexport) int SetAngularControl();

extern "C" __declspec(dllexport) int SetCartesianControl();

extern "C" __declspec(dllexport) int StartControlAPI();

extern "C" __declspec(dllexport) int StopControlAPI();

extern "C" __declspec(dllexport) int RestoreFactoryDefault();

extern "C" __declspec(dllexport) int SendJoystickCommand(JoystickCommand joystickCommand);

extern "C" __declspec(dllexport) int SendAdvanceTrajectory(TrajectoryPoint trajectory);

extern "C" __declspec(dllexport) int SendBasicTrajectory(TrajectoryPoint trajectory);

extern "C" __declspec(dllexport) int GetClientConfigurations(ClientConfigurations &config);

extern "C" __declspec(dllexport) int GetAllRobotIdentity(RobotIdentity robotIdentity[MAX_KINOVA_DEVICE], int & count);

extern "C" __declspec(dllexport) int GetRobotIdentity(RobotIdentity &robotIdentity);

extern "C" __declspec(dllexport) int SetClientConfigurations(ClientConfigurations config);

extern "C" __declspec(dllexport) int EraseAllTrajectories();

extern "C" __declspec(dllexport) int GetPositionCurrentActuators(float Response[POSITION_CURRENT_COUNT]);

extern "C" __declspec(dllexport) int SetActuatorPID(unsigned int address, float P, float I, float D);

extern "C" __declspec(dllexport) int GetAngularCommand(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetCartesianCommand(CartesianPosition &Response);

extern "C" __declspec(dllexport) int GetAngularCurrentMotor(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetAngularVelocity(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetControlType(int &Response);

extern "C" __declspec(dllexport) int StartForceControl();

extern "C" __declspec(dllexport) int StopForceControl();

extern "C" __declspec(dllexport) int StartRedundantJointNullSpaceMotion();

extern "C" __declspec(dllexport) int StopRedundantJointNullSpaceMotion();

extern "C" __declspec(dllexport) int ActivateExtraProtectionPinchingWrist(int state);

extern "C" __declspec(dllexport) int ActivateCollisionAutomaticAvoidance(int state); //not available on Jaco, Jaco Spherical 6 DOF and Mico models. 

extern "C" __declspec(dllexport) int ActivateSingularityAutomaticAvoidance(int state); //not available on Jaco, Jaco Spherical 6 DOF and Mico models. 

extern "C" __declspec(dllexport) int ActivateAutoNullSpaceMotionCartesian(int state); //not available on Jaco, Jaco Spherical 6 DOF and Mico models. 

extern "C" __declspec(dllexport) int StartCurrentLimitation();

extern "C" __declspec(dllexport) int StopCurrentLimitation();

extern "C" __declspec(dllexport) int GetSystemErrorCount(unsigned int &Response);

extern "C" __declspec(dllexport) int GetSystemError(unsigned int indexError, SystemError &Response);

extern "C" __declspec(dllexport) int ClearErrorLog();

extern "C" __declspec(dllexport) int EraseAllProtectionZones();

//Internal use only
extern "C" __declspec(dllexport) int SetSerialNumber(char Command[STRING_LENGTH], char temp[STRING_LENGTH]);

extern "C" __declspec(dllexport) int SetDefaultGravityParam(float Command[GRAVITY_PARAM_SIZE]);

extern "C" __declspec(dllexport) int GetControlMapping(ControlMappingCharts &Response);

extern "C" __declspec(dllexport) int GetProtectionZone(ZoneList &Response);

extern "C" __declspec(dllexport) int SetProtectionZone(ZoneList Command);

extern "C" __declspec(dllexport) int GetGripperStatus(Gripper &Response);

extern "C" __declspec(dllexport) int GetQuickStatus(QuickStatus &Response);

extern "C" __declspec(dllexport) int GetForcesInfo(ForcesInfo &Response);

extern "C" __declspec(dllexport) int SetControlMapping(ControlMappingCharts Command);

extern "C" __declspec(dllexport) int ProgramFlash(const char * filename);

extern "C" __declspec(dllexport) int SetJointZero(int ActuatorAdress);

extern "C" __declspec(dllexport) int SetTorqueZero(int ActuatorAdress);

extern "C" __declspec(dllexport) int SetTorqueGain(int ActuatorAdress, float Gain);

extern "C" __declspec(dllexport) int SetActuatorPIDFilter(int ActuatorAdress, float filterP, float filterI, float filterD);

extern "C" __declspec(dllexport) int SetActuatorAddress(int ActuatorAdress, int newAddress);

extern "C" __declspec(dllexport) int GetGeneralInformations(GeneralInformations &Response);

extern "C" __declspec(dllexport) int SetFrameType(int frameType);

extern "C" __declspec(dllexport) int SetCartesianForceMinMax(CartesianInfo min, CartesianInfo max);

extern "C" __declspec(dllexport) int SetCartesianInertiaDamping(CartesianInfo inertia, CartesianInfo damping);

extern "C" __declspec(dllexport) int SetAngularTorqueMinMax(AngularInfo min, AngularInfo max);

extern "C" __declspec(dllexport) int SetAngularInertiaDamping(AngularInfo inertia, AngularInfo damping);

//Internal use only
extern "C" __declspec(dllexport) int SetDevValue(std::vector<float> command);

//Internal use only
extern "C" __declspec(dllexport) int GetDevValue(std::vector<float> &Response);

extern "C" __declspec(dllexport) int SetSpasmFilterValues(float Command[SPASM_FILTER_COUNT], int activationStatus);

extern "C" __declspec(dllexport) int GetSpasmFilterValues(float Response[SPASM_FILTER_COUNT], int &activationStatus);

extern "C" __declspec(dllexport) int MoveHome();

extern "C" __declspec(dllexport) int GetAngularForceGravityFree(AngularPosition &Response);

extern "C" __declspec(dllexport) int GetActuatorAcceleration(AngularAcceleration &Response);

extern "C" __declspec(dllexport) int InitFingers();

extern "C" __declspec(dllexport) int GetPeripheralInventory(PeripheralInfo list[MAX_INVENTORY]);

//Internal use only
extern "C" __declspec(dllexport) int SetModel(char Command[STRING_LENGTH], char temp[STRING_LENGTH]);

extern "C" __declspec(dllexport) int GetJoystickValue(JoystickCommand &joystickCommand);

extern "C" __declspec(dllexport) int SetRobotConfiguration(int ConfigID);

extern "C" __declspec(dllexport) int GetCommandVelocity(float cartesianVelocity[CARTESIAN_SIZE], float angularVelocity[MAX_ACTUATORS]);

extern "C" __declspec(dllexport) int GetEndEffectorOffset(unsigned int &status, float &x, float &y, float &z);

extern "C" __declspec(dllexport) int SetEndEffectorOffset(unsigned int status, float x, float y, float z);



extern "C" __declspec(dllexport) int SendAngularTorqueCommand(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SendCartesianForceCommand(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueActuatorGain(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueActuatorDamping(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SwitchTrajectoryTorque(GENERALCONTROL_TYPE type);

extern "C" __declspec(dllexport) int SetTorqueCommandMax(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueSafetyFactor(float factor);

extern "C" __declspec(dllexport) int SetTorqueGainMax(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueRateLimiter(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueFeedCurrent(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueFeedVelocity(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorquePositionLimitDampingGain(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorquePositionLimitDampingMax(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorquePositionLimitRepulsGain(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorquePositionLimitRepulsMax(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueFilterVelocity(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueFilterMeasuredTorque(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueFilterError(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueFilterControlEffort(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetGravityType(GRAVITY_TYPE type);

extern "C" __declspec(dllexport) int SetGravityVector(float gravityVector[GRAVITY_VECTOR_SIZE]);

extern "C" __declspec(dllexport) int SetGravityOptimalZParam(float Command[GRAVITY_PARAM_SIZE]);

extern "C" __declspec(dllexport) int SetGravityManualInputParam(float Command[GRAVITY_PARAM_SIZE]);

extern "C" __declspec(dllexport) int GetAngularTorqueCommand(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int GetAngularTorqueGravityEstimation(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetActuatorMaxVelocity(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetSwitchThreshold(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetPositionLimitDistance(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueControlType(TORQUECONTROL_TYPE type);

extern "C" __declspec(dllexport) int SetGravityPayload(float Command[GRAVITY_PAYLOAD_SIZE]);

extern "C" __declspec(dllexport) int SetTorqueVibrationController(float activationStatus);

extern "C" __declspec(dllexport) int SetTorqueRobotProtection(int protectionLevel);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueVelocityLimitFilter(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueFeedFilter(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueStaticFriction(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueErrorDeadband(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueBrake(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueInactivityTimeActuator(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueInactivityTimeMainController(int time);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueDampingMax(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueFeedVelocityUnderGain(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueFeedCurrentVoltage(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueStaticFrictionMax(float Command[COMMAND_SIZE]);

//Internal use only
extern "C" __declspec(dllexport) int SetTorqueErrorResend(float Command[COMMAND_SIZE]);

extern "C" __declspec(dllexport) int RunGravityZEstimationSequence(ROBOT_TYPE type, float OptimalzParam[OPTIMAL_Z_PARAM_SIZE]);

extern "C" __declspec(dllexport) int RunGravityZEstimationSequence7DOF(ROBOT_TYPE type, float OptimalzParam[OPTIMAL_Z_PARAM_SIZE_7DOF]);

extern "C" __declspec(dllexport) int GetTrajectoryTorqueMode(int&);

extern "C" __declspec(dllexport) int SetTorqueInactivityType(int);

//NEW ETHERNET EXPORTED FUNCTIONS
extern "C" __declspec(dllexport) int SetEthernetConfiguration(EthernetConfiguration * config);

extern "C" __declspec(dllexport) int GetEthernetConfiguration(EthernetConfiguration * config);

//DO NOT USE only for Kinova
extern "C" __declspec(dllexport) int SetLocalMACAddress(unsigned char mac[MAC_ADDRESS_LENGTH], char temp[STRING_LENGTH]);
