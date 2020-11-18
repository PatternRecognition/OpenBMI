
#include "JacoSDKWrapper.h"

#include "mex.h"


using namespace std;

//A handle to the API.
HINSTANCE commandLayer_handle = NULL;

//Function pointers to the functions we need
int(*MyInitAPI)();
int(*MyCloseAPI)();
int(*MyGetGeneralInformations)(GeneralInformations &Response);
int(*MyGetDevices)(KinovaDevice devices[MAX_KINOVA_DEVICE], int &result);
int(*MySetActiveDevice)(KinovaDevice device);

//int(*MyGetAngularCommand)(AngularPosition &);
int(*MyGetAngularPosition)(AngularPosition &);
int (*MyGetAngularCommand)(AngularPosition &);
int(*MyGetAngularVelocity)(AngularPosition &Response);
int(*MyGetAngularForce)(AngularPosition &Response);
int(*MyGetCartesianPosition)(CartesianPosition &);
int (*MyGetCartesianCommand)(CartesianPosition &);
int (*MyGetCartesianForce)(CartesianPosition &);
int (*MyMoveHome)();
int(*MySetTorqueSafetyFactor)(float factor);
int(*MySetTorqueVibrationController)(float value);
int(*MySwitchTrajectoryTorque)(GENERALCONTROL_TYPE);
int(*MySetTorqueControlType)(TORQUECONTROL_TYPE type);
int(*MyRunGravityZEstimationSequence)(ROBOT_TYPE type, double OptimalzParam[OPTIMAL_Z_PARAM_SIZE]);
int(*MySetGravityOptimalZParam)(float Command[GRAVITY_PARAM_SIZE]);
int(*MySetGravityType)(GRAVITY_TYPE Type);
int (*MyInitFingers)();
int(*MySendBasicTrajectory)(TrajectoryPoint command);
int(*MySendAngularTorqueCommand)(float Command[COMMAND_SIZE]);
int (*MyGetCodeVersion)(int Response[CODE_VERSION_COUNT]);
int (*MyStartForceControl)();
int (*MyStopForceControl)();
int (*MyGetEndEffectorOffset)(unsigned int*, float*, float*, float*);
int (*MySetEndEffectorOffset)(unsigned int status, float x, float y, float z);
int (*MyGetProtectionZone)(ZoneList &Response);
int (*MyEraseAllProtectionZones)();
int (*MySetProtectionZone)(ZoneList Command);
int (*MySetCartesianControl)();
int (*MyGetGlobalTrajectoryInfo)(TrajectoryFIFO &Response);
int (*MySendAdvanceTrajectory)(TrajectoryPoint command);
int (*MySetPositionLimitDistance)(float Command[COMMAND_SIZE]);
int (*MySetActuatorPID)(unsigned int address, float P, float I, float D);
int (*MyRefresDevicesList)();





KinovaDevice list[MAX_KINOVA_DEVICE];
int devicesCount;


bool openKinovaLibrary()
{
    int result;
    mexPrintf("Loading library..."); 
    if(commandLayer_handle == NULL )
    {
        mexPrintf("Loading library..."); 
        commandLayer_handle = LoadLibrary("CommandLayerWindows.dll");
//         For Ethernet control choose CommandLayerEthernet.dll
//         commandLayer_handle = LoadLibrary("CommandLayerEthernet.dll");
        if(commandLayer_handle == NULL )
        {
            mexPrintf("Failed to load library.\n");
            return false;
        }
        mexPrintf(" Success\n");
    }
    else
    {
        mexPrintf("Library is already loaded.\n");
    }
    
    if (MyInitAPI != NULL)
    {
        mexPrintf("API is already open.\n");
        return true;
    }
    
    // Try to open API
    mexPrintf("Initializating APIs...");

    //We load the functions from the library (Under Windows, use GetProcAddress)
    MyInitAPI = (int(*)()) GetProcAddress(commandLayer_handle, "InitAPI");
    MyCloseAPI = (int(*)()) GetProcAddress(commandLayer_handle, "CloseAPI");
    MyGetGeneralInformations = (int(*)(GeneralInformations &info)) GetProcAddress(commandLayer_handle, "GetGeneralInformations");

    MyGetDevices = (int(*)(KinovaDevice devices[MAX_KINOVA_DEVICE], int &result)) GetProcAddress(commandLayer_handle, "GetDevices");
    MySetActiveDevice = (int(*)(KinovaDevice devices)) GetProcAddress(commandLayer_handle, "SetActiveDevice");
    MyGetAngularCommand = (int (*)(AngularPosition &)) GetProcAddress(commandLayer_handle,"GetAngularCommand");
    MyGetAngularPosition = (int(*)(AngularPosition &)) GetProcAddress(commandLayer_handle, "GetAngularPosition");
    MyGetAngularVelocity = (int(*)(AngularPosition &)) GetProcAddress(commandLayer_handle, "GetAngularVelocity");
    MyGetAngularForce = (int(*)(AngularPosition &Response)) GetProcAddress(commandLayer_handle, "GetAngularForce");
  	MyGetCartesianPosition = (int(*)(CartesianPosition &)) GetProcAddress(commandLayer_handle, "GetCartesianPosition");
    MyGetCartesianCommand = (int(*)(CartesianPosition &)) GetProcAddress(commandLayer_handle, "GetCartesianCommand");
    MyGetCartesianForce = (int (*)(CartesianPosition &)) GetProcAddress(commandLayer_handle,"GetCartesianForce");
    MyMoveHome = (int (*)()) GetProcAddress(commandLayer_handle,"MoveHome");
    
    MySetTorqueSafetyFactor = (int(*)(float)) GetProcAddress(commandLayer_handle, "SetTorqueSafetyFactor");
    MySetTorqueVibrationController = (int(*)(float)) GetProcAddress(commandLayer_handle, "SetTorqueVibrationController");
    MySwitchTrajectoryTorque = (int(*)(GENERALCONTROL_TYPE)) GetProcAddress(commandLayer_handle, "SwitchTrajectoryTorque");
    MySetTorqueControlType = (int(*)(TORQUECONTROL_TYPE)) GetProcAddress(commandLayer_handle, "SetTorqueControlType");
    MyRunGravityZEstimationSequence = (int(*)(ROBOT_TYPE, double OptimalzParam[OPTIMAL_Z_PARAM_SIZE])) GetProcAddress(commandLayer_handle, "RunGravityZEstimationSequence");
    MySetGravityOptimalZParam = (int(*)(float Command[GRAVITY_PARAM_SIZE])) GetProcAddress(commandLayer_handle, "SetGravityOptimalZParam");
    MySetGravityType = (int(*)(GRAVITY_TYPE Type)) GetProcAddress(commandLayer_handle, "SetGravityType");
    MyInitFingers = (int (*)()) GetProcAddress(commandLayer_handle,"InitFingers");
  	MySendBasicTrajectory = (int(*)(TrajectoryPoint)) GetProcAddress(commandLayer_handle, "SendBasicTrajectory");
  	MySendAngularTorqueCommand = (int(*)(float Command[COMMAND_SIZE])) GetProcAddress(commandLayer_handle, "SendAngularTorqueCommand");
	MyGetCodeVersion = (int(*) (int Response[CODE_VERSION_COUNT])) GetProcAddress(commandLayer_handle, "GetCodeVersion"); 
    MyStartForceControl = (int(*)()) GetProcAddress(commandLayer_handle, "StartForceControl");
    MyStopForceControl = (int(*)()) GetProcAddress(commandLayer_handle, "StopForceControl");
    MyGetEndEffectorOffset = (int(*)(unsigned int*, float*, float*, float*)) GetProcAddress(commandLayer_handle, "GetEndEffectorOffset");
    MySetEndEffectorOffset = (int(*)(unsigned int, float, float, float)) GetProcAddress(commandLayer_handle, "SetEndEffectorOffset");
    MyGetProtectionZone = (int(*)(ZoneList &)) GetProcAddress(commandLayer_handle, "GetProtectionZone");
    MyEraseAllProtectionZones = (int(*)()) GetProcAddress(commandLayer_handle, "EraseAllProtectionZones");
    MySetProtectionZone = (int(*)(ZoneList)) GetProcAddress(commandLayer_handle, "SetProtectionZone");
    MySetCartesianControl = (int(*)()) GetProcAddress(commandLayer_handle, "SetCartesianControl");
    MyGetGlobalTrajectoryInfo = (int(*)(TrajectoryFIFO &Response)) GetProcAddress(commandLayer_handle, "GetGlobalTrajectoryInfo");
    MySendAdvanceTrajectory = (int(*)(TrajectoryPoint)) GetProcAddress(commandLayer_handle, "SendAdvanceTrajectory");
    MySetPositionLimitDistance = (int(*)(float Command[COMMAND_SIZE])) GetProcAddress(commandLayer_handle, "SetPositionLimitDistance");
    MySetActuatorPID = (int(*)(unsigned int, float, float, float)) GetProcAddress(commandLayer_handle, "SetActuatorPID");
    MyGetGlobalTrajectoryInfo = (int(*)(TrajectoryFIFO &Response)) GetProcAddress(commandLayer_handle, "GetGlobalTrajectoryInfo");
	MyRefresDevicesList = (int(*)()) GetProcAddress(commandLayer_handle, "RefresDevicesList");



    
    if (MyInitAPI == NULL || MyCloseAPI == NULL || MyGetGeneralInformations == NULL || MyMoveHome == NULL)        
    {
        mexPrintf("* * *  E R R O R   D U R I N G   I N I T I A L I Z A T I O N  * * *\n");
        return false;
    }
    else
    {
        mexPrintf(" Success\n");

        result = (*MyInitAPI)();
		(*MyRefresDevicesList)();

        //mexPrintf("Initialization's result: %d\n", result);           

        devicesCount = MyGetDevices(list, result);

        //cout << "Found a robot on the USB bus (" << list[i].SerialNumber << ")" << endl;

        mexPrintf("Verifying number of devices...");
        if (devicesCount == 0)
        {
            mexPrintf(" No devices found\n");
            return false;
        }
        else if (devicesCount > 1)
        {
            mexPrintf(" More than one device found. Not supported\n");
            return false;                      
        }
        else
        {
             mexPrintf(" Success: one device\n");
             mexPrintf("Device type: %d\n", list[0].DeviceType);
             // JACOV1_ASSISTIVE = 0, 
             // MICO_6DOF_SERVICE = 1, 
             // MICO_4DOF_SERVICE = 2, 
             // JACOV2_6DOF_SERVICE = 3, 
             // JACOV2_4DOF_SERVICE = 4, 
             // MICO_6DOF_ASSISTIVE = 5, 
             // JACOV2_6DOF_ASSISTIVE = 6,
             // SPHERICAL_6DOF_SERVICE = 7,
             // SPHERICAL_7DOF_SERVICE = 8
             return true;
        }    
    }
    
}


bool closeKinovaLibrary()
{
    int result;
    if (MyInitAPI != NULL)
    {
        mexPrintf("Closing   API\n");
        result = (*MyCloseAPI)();
        MyInitAPI = NULL;
    }
    else
    {
        mexPrintf("API is not open\n"); 
    }
    if(commandLayer_handle != NULL )
    {
        mexPrintf("Closing library\n");
        FreeLibrary(commandLayer_handle);
        commandLayer_handle = NULL;
    }
    else
    {
         mexPrintf("Library is not loaded\n");   
    }
    return true;
}


/* The computational routine */
// arrayProduct(double x, double *y, double *z, mwSize n)
bool getJointsPosition(double *pos)
{    
    AngularPosition dataPosition;

    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetAngularPosition != NULL)
    {
        for (int i = 0; i < devicesCount; i++)
        {

            //Setting the current device as the active device.
            MySetActiveDevice(list[i]);

            (*MyGetAngularPosition)(dataPosition);                

            pos[0] =  kDeg2rad * dataPosition.Actuators.Actuator1;
            pos[1] =  kDeg2rad * dataPosition.Actuators.Actuator2;
            pos[2] =  kDeg2rad * dataPosition.Actuators.Actuator3;
            pos[3] =  kDeg2rad * dataPosition.Actuators.Actuator4;
            pos[4] =  kDeg2rad * dataPosition.Actuators.Actuator5;
            pos[5] =  kDeg2rad * dataPosition.Actuators.Actuator6;
			pos[6] =  kDeg2rad * dataPosition.Actuators.Actuator7;
            
            pos[7] = dataPosition.Fingers.Finger1;
            pos[8] = dataPosition.Fingers.Finger2;
            pos[9] = dataPosition.Fingers.Finger3;         
        }
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }            
}

bool getJointsVelocity(double *vel)
{    
    AngularPosition angularVelocity;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetAngularVelocity != NULL)
    {
        for (int i = 0; i < devicesCount; i++)
        {
            MySetActiveDevice(list[i]);
            MyGetAngularVelocity(angularVelocity);                
            vel[0] = kDeg2rad * angularVelocity.Actuators.Actuator1;
            vel[1] = kDeg2rad * angularVelocity.Actuators.Actuator2;
            vel[2] = kDeg2rad * angularVelocity.Actuators.Actuator3;
            vel[3] = kDeg2rad * angularVelocity.Actuators.Actuator4;
            vel[4] = kDeg2rad * angularVelocity.Actuators.Actuator5;
            vel[5] = kDeg2rad * angularVelocity.Actuators.Actuator6;
			vel[6] = kDeg2rad * angularVelocity.Actuators.Actuator7;
            
            vel[7] = angularVelocity.Fingers.Finger1;
            vel[8] = angularVelocity.Fingers.Finger2;
            vel[9] = angularVelocity.Fingers.Finger3;
        }
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }            
}

bool getJointsTorque(double *torque)
{    
    AngularPosition angularForce;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetAngularForce != NULL)
    {
        for (int i = 0; i < devicesCount; i++)
        {
            MySetActiveDevice(list[i]);
            MyGetAngularForce(angularForce);                
            torque[0] = angularForce.Actuators.Actuator1;
            torque[1] = angularForce.Actuators.Actuator2;
            torque[2] = angularForce.Actuators.Actuator3;
            torque[3] = angularForce.Actuators.Actuator4;
            torque[4] = angularForce.Actuators.Actuator5;
            torque[5] = angularForce.Actuators.Actuator6;
			torque[6] = angularForce.Actuators.Actuator7;
            
            torque[7] = angularForce.Fingers.Finger1;
            torque[8] = angularForce.Fingers.Finger2;
            torque[9] = angularForce.Fingers.Finger3;
        }
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }            
}


bool getJointsTemperature(double *temp)
{    
  	GeneralInformations data;

    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetGeneralInformations !=NULL)
    {
        for (int i = 0; i < devicesCount; i++)
        {
            MySetActiveDevice(list[i]);
            MyGetGeneralInformations(data);                
            temp[0] = data.ActuatorsTemperatures[0];
            temp[1] = data.ActuatorsTemperatures[1];
            temp[2] = data.ActuatorsTemperatures[2];
            temp[3] = data.ActuatorsTemperatures[3];
            temp[4] = data.ActuatorsTemperatures[4];
            temp[5] = data.ActuatorsTemperatures[5];
			temp[6] = data.ActuatorsTemperatures[6];
            
            temp[7] = data.FingersTemperatures[0];
            temp[8] = data.FingersTemperatures[1];
            temp[9] = data.FingersTemperatures[2];
        }
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }            
}


bool getEndEffectorPose(double *pose)
{    
    CartesianPosition dataPosition;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetCartesianPosition != NULL)
    {
        for (int i = 0; i < devicesCount; i++)
        {
            MySetActiveDevice(list[i]);
            (*MyGetCartesianPosition)(dataPosition);

            pose[0] = dataPosition.Coordinates.X;
            pose[1] = dataPosition.Coordinates.Y;
            pose[2] = dataPosition.Coordinates.Z;
            pose[3] = dataPosition.Coordinates.ThetaX;
            pose[4] = dataPosition.Coordinates.ThetaY;
            pose[5] = dataPosition.Coordinates.ThetaZ;           

        }
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}


bool getEndEffectorWrench(double *wrench)
{    
    CartesianPosition data;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetCartesianForce!=NULL)
    {
        for (int i = 0; i < devicesCount; i++)
        {
            MySetActiveDevice(list[i]);
            MyGetCartesianForce(data);

            wrench[0] = data.Coordinates.X;
            wrench[1] = data.Coordinates.Y;
            wrench[2] = data.Coordinates.Z;
            wrench[3] = data.Coordinates.ThetaX;
            wrench[4] = data.Coordinates.ThetaY;
            wrench[5] = data.Coordinates.ThetaZ;         
        }
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}


bool moveToHomePosition(void)
{
    int data = 1;
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyMoveHome != NULL)
    {
        MyMoveHome();
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}

bool setPositionControlMode(void)
{
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySwitchTrajectoryTorque != NULL)
    {
        MySwitchTrajectoryTorque(POSITION);
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}
// MySwitchTorque(POSITION);

bool setDirectTorqueControlMode(void)
{
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySwitchTrajectoryTorque != NULL
            && MySetTorqueVibrationController !=NULL && MySetTorqueSafetyFactor != NULL
            && MySetTorqueControlType != NULL)
    {
        MySetTorqueControlType(DIRECTTORQUE);
        // Set the safety factor to 0.8
        MySetTorqueSafetyFactor(0.8f);
        // Set the vibration controller to 0.5
        MySetTorqueVibrationController(0.5);
        
        // Switch to torque control
        // (Here we switch before sending torques. 
        // The switch is possible because the gravity torques are already taken into account.)
        MySwitchTrajectoryTorque(TORQUE);
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }  
}
//MySwitchTorque(TORQUE) == MySwitchTrajectoryTorque;


bool runGravityCalibration(void)
{
    double OptimalzParam[OPTIMAL_Z_PARAM_SIZE];
    float GravParamCommand[GRAVITY_PARAM_SIZE];
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyRunGravityZEstimationSequence != NULL
            && MySetGravityOptimalZParam != NULL && MySetGravityType !=NULL)
    {
        // Choose robot type
        ROBOT_TYPE type = JACOV2_6DOF_SERVICE;

        // Run identification sequence
        // CAUTION READ THE FUNCTION DOCUMENTATION BEFORE
        MyRunGravityZEstimationSequence(type, OptimalzParam);
        
        //send the optimal gravity parameters to the robot
        for (int pp = 0; pp<OPTIMAL_Z_PARAM_SIZE; pp++)
        {
            GravParamCommand[pp] = (float) OptimalzParam[pp];
        }
        MySetGravityOptimalZParam(GravParamCommand);

        //set optimal gravity type
        MySetGravityType(OPTIMAL);
        
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }  
}


bool initializeFingers(void)
{
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyInitFingers != NULL)
    {
        MyInitFingers();
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}


bool sendJointPositions(double *pos)
{
    TrajectoryPoint pointToSend;
    AngularPosition data;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySendBasicTrajectory != NULL)
    {
        pointToSend.Position.Type = ANGULAR_POSITION;
        // new
        pointToSend.Position.HandMode = POSITION_MODE;

        
        
        // We need to convert to degrees for Kinova API
        pointToSend.Position.Actuators.Actuator1 = kRad2deg * pos[0];
        pointToSend.Position.Actuators.Actuator2 = kRad2deg * pos[1];
        pointToSend.Position.Actuators.Actuator3 = kRad2deg * pos[2];
        pointToSend.Position.Actuators.Actuator4 = kRad2deg * pos[3];
        pointToSend.Position.Actuators.Actuator5 = kRad2deg * pos[4];
        pointToSend.Position.Actuators.Actuator6 = kRad2deg * pos[5];
        pointToSend.Position.Actuators.Actuator7 = kRad2deg * pos[6];
        
        // This is required otherwise it only works for the first time 
        // you send the command unless you move the joystick. 
        // pointToSend.Position.Fingers.Finger1 = 0;
		// pointToSend.Position.Fingers.Finger2 = 0;
		// pointToSend.Position.Fingers.Finger3 = 0;
        
        
        
        // new Get current finger commands and set them in the point to send cmd
        int result = NO_ERROR_KINOVA;
        result = MyGetAngularCommand(data);
        pointToSend.Position.Fingers.Finger1 = data.Fingers.Finger1; 
        pointToSend.Position.Fingers.Finger2  = data.Fingers.Finger2; 
        pointToSend.Position.Fingers.Finger3  = data.Fingers.Finger3; 
        //pointToSend.Position.Actuators.Actuator4 = data.Actuators.Actuator4; 
        //pointToSend.Position.Actuators.Actuator5 = data.Actuators.Actuator5; 
        //pointToSend.Position.Actuators.Actuator6 = data.Actuators.Actuator6; 
        
        
        if (result == NO_ERROR_KINOVA) {
            MySendBasicTrajectory(pointToSend);
            return true;
        }
        else
        {
            mexPrintf("Could not get angular position\n");
            return false;
        }        
        
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}


bool sendJointAndFingerPositions(double *jpos, double *fpos)
{
    TrajectoryPoint pointToSend;
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySendBasicTrajectory != NULL)
    {
        pointToSend.Position.HandMode = POSITION_MODE;
        pointToSend.Position.Type = ANGULAR_POSITION;
        
        // Send finger positions
        pointToSend.Position.Fingers.Finger1 = fpos[0];
		pointToSend.Position.Fingers.Finger2 = fpos[1];
		pointToSend.Position.Fingers.Finger3 = fpos[2]; 
        
        // We need to convert to degrees for Kinova API
        pointToSend.Position.Actuators.Actuator1 = kRad2deg * jpos[0];
        pointToSend.Position.Actuators.Actuator2 = kRad2deg * jpos[1];
        pointToSend.Position.Actuators.Actuator3 = kRad2deg * jpos[2];
        pointToSend.Position.Actuators.Actuator4 = kRad2deg * jpos[3];
        pointToSend.Position.Actuators.Actuator5 = kRad2deg * jpos[4];
        pointToSend.Position.Actuators.Actuator6 = kRad2deg * jpos[5];
        pointToSend.Position.Actuators.Actuator7 = kRad2deg * jpos[6];
        

        
        MySendBasicTrajectory(pointToSend);
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}

//MySendBasicTrajectory(pointToSend);


bool sendJointVelocities(double *vel)
{
    TrajectoryPoint pointToSend;    
    if (MyInitAPI != NULL && commandLayer_handle != NULL)
    {
        pointToSend.Position.Type = ANGULAR_VELOCITY;   
        // We need to convert to degrees for Kinova API
        pointToSend.Position.Actuators.Actuator1 = kRad2deg * vel[0];
        pointToSend.Position.Actuators.Actuator2 = kRad2deg * vel[1];
        pointToSend.Position.Actuators.Actuator3 = kRad2deg * vel[2];
        pointToSend.Position.Actuators.Actuator4 = kRad2deg * vel[3];
        pointToSend.Position.Actuators.Actuator5 = kRad2deg * vel[4];
        pointToSend.Position.Actuators.Actuator6 = kRad2deg * vel[5];
        pointToSend.Position.Actuators.Actuator7 = kRad2deg * vel[6];
        
        MySendBasicTrajectory(pointToSend);

        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}
//MySendBasicTrajectory(pointToSend);


bool sendJointTorques(double *torque)
{
    float TorqueCommand[COMMAND_SIZE]; 
    
    for (int i = 0; i<COMMAND_SIZE; i++)
    {
        TorqueCommand[i] = 0.0; //initialization
    }
    
    //mexPrintf("Command size %d\n",COMMAND_SIZE);
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySendAngularTorqueCommand != NULL)
    {
        TorqueCommand[0] = torque[0];
        TorqueCommand[1] = torque[1];
        TorqueCommand[2] = torque[2];
        TorqueCommand[3] = torque[3];
        TorqueCommand[4] = torque[4];
        TorqueCommand[5] = torque[5];
        TorqueCommand[6] = torque[6];
        MySendAngularTorqueCommand(TorqueCommand);
        return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}
//MySendAngularTorqueCommand(TorqueCommand);


bool sendFingerPositions(double *pos)
{
    TrajectoryPoint pointToSend;
    AngularPosition data;

    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySendBasicTrajectory != NULL
            && MyGetAngularCommand != NULL)
    {
        // Set finger positions
        pointToSend.Position.HandMode = POSITION_MODE;
        pointToSend.Position.Type = ANGULAR_POSITION;           
        pointToSend.Position.Fingers.Finger1 = pos[0];
		pointToSend.Position.Fingers.Finger2 = pos[1];
		pointToSend.Position.Fingers.Finger3 = pos[2];       
        
        // Get current joint commands and set them in the point to send cmd
        int result = NO_ERROR_KINOVA;
        result = MyGetAngularCommand(data);
        pointToSend.Position.Actuators.Actuator1 = data.Actuators.Actuator1; 
        pointToSend.Position.Actuators.Actuator2 = data.Actuators.Actuator2; 
        pointToSend.Position.Actuators.Actuator3 = data.Actuators.Actuator3; 
        pointToSend.Position.Actuators.Actuator4 = data.Actuators.Actuator4; 
        pointToSend.Position.Actuators.Actuator5 = data.Actuators.Actuator5; 
        pointToSend.Position.Actuators.Actuator6 = data.Actuators.Actuator6; 
        pointToSend.Position.Actuators.Actuator7 = data.Actuators.Actuator7; 
        
        if (result == NO_ERROR_KINOVA) {
            MySendBasicTrajectory(pointToSend);
            return true;
        }
        else
        {
            mexPrintf("Could not get angular position\n");
            return false;
        }
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}

bool sendCartesianPositions(double *pos)
{
    TrajectoryPoint pointToSend;
    CartesianPosition data;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySendAdvanceTrajectory != NULL)
    {
        pointToSend.Position.Type = CARTESIAN_POSITION;
        MyGetCartesianCommand(data);
       // Set cartesian position     
        pointToSend.Position.CartesianPosition.X = data.Coordinates.X + pos[0];
    	pointToSend.Position.CartesianPosition.Y = data.Coordinates.Y + pos[1];
		pointToSend.Position.CartesianPosition.Z = data.Coordinates.Z + pos[2];
		pointToSend.Position.CartesianPosition.ThetaX = data.Coordinates.ThetaX + pos[3];
		pointToSend.Position.CartesianPosition.ThetaY = data.Coordinates.ThetaY + pos[4];
		pointToSend.Position.CartesianPosition.ThetaZ = data.Coordinates.ThetaZ + pos[5];
       
            
       MySendAdvanceTrajectory(pointToSend);
       return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}

bool sendCartesianVelocity(double *vel)
{
    TrajectoryPoint pointToSend;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySendAdvanceTrajectory != NULL)
    {
       pointToSend.Position.Type = CARTESIAN_VELOCITY;
       pointToSend.Position.HandMode = HAND_NOMOVEMENT;
       // Set cartesian velocity Cmd
       pointToSend.Position.CartesianPosition.X = vel[0];
       pointToSend.Position.CartesianPosition.Y = vel[1];
       pointToSend.Position.CartesianPosition.Z = vel[2];
       pointToSend.Position.CartesianPosition.ThetaX = kRad2deg * vel[3];
       pointToSend.Position.CartesianPosition.ThetaY = kRad2deg * vel[4];
       pointToSend.Position.CartesianPosition.ThetaZ = kRad2deg * vel[5];
       MySendAdvanceTrajectory(pointToSend);
       return true;
    }
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    } 
}

bool GetDOF (double *DOF)
{
    // JACOV1_ASSISTIVE =       0, 
    // MICO_6DOF_SERVICE =      1, 
    // MICO_4DOF_SERVICE =      2, 
    // JACOV2_6DOF_SERVICE =    3, 
    // JACOV2_4DOF_SERVICE =    4, 
    // MICO_6DOF_ASSISTIVE =    5, 
    // JACOV2_6DOF_ASSISTIVE =  6,
    // SPHERICAL_6DOF_SERVICE = 7,
    // SPHERICAL_7DOF_SERVICE = 8   
            
    int Type;
    
    if (MyInitAPI != NULL && commandLayer_handle != NULL)
    {
        Type = list[0].DeviceType;
        if(Type == 2 || Type == 4)
        {
            *DOF = 4;
        }
        else if(Type == 0 || Type == 1 || Type == 3 || Type == 5 || Type == 6 || Type == 7)
        {
            *DOF = 6;
        }
        else if(Type == 8)
        {
            *DOF = 7;
        }
        else
        {
            mexPrintf("Can't get robot type\n");
            return false;   
        }
        return true;
    }
    
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }           
}
bool startForceControl()
{
     if (MyInitAPI != NULL && commandLayer_handle != NULL && MyStartForceControl != NULL)
     {
        MyStartForceControl();  
        return true;
     }
     
    else
    {
        mexPrintf("Library or API not open\n");
        return false;
    }

}


bool stopForceControl()
{
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyStopForceControl != NULL)
    {
        MyStopForceControl();  
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        return false;
    }
}

bool getEndEffectorOffset(double *offset)
{   
    unsigned int status;
    float x,y,z;
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetEndEffectorOffset != NULL)
    {   
        // Get end effector offset
        MyGetEndEffectorOffset(&status, &x, &y, &z);
        offset[0] = status;
        offset[1] = x;
        offset[2] = y;
        offset[3] = z;
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        return false;
    }
}

bool setEndEffectorOffset(double *offset)
{
    unsigned int status;
    float x;
    float y;
    float z;
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySetEndEffectorOffset != NULL)
    {   
        // Set end effector offset
        status = offset[0];
        x = offset[1];
        y = offset[2];
        z = offset[3];
        MySetEndEffectorOffset(status, x, y, z);
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        return false;
    }
}

bool getProtectionZone(double *zone)
{
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetProtectionZone != NULL)
    {
        ZoneList zoneList;
        MyGetProtectionZone(zoneList);
        *zone = zoneList.NbZones;
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        return false;
    }
}

bool eraseAllProtectionZones()
{
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyEraseAllProtectionZones != NULL)
    {
        MyEraseAllProtectionZones();
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        
        return false;
    }
}

bool setProtectionZone(double *zone)
{int result;
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MySetProtectionZone != NULL)
    {   
        ZoneList zones;
        ZoneList zoneList;
        MyGetProtectionZone(zoneList);
       
        zones.NbZones = zoneList.NbZones + 1;
        zones.Zones[zones.NbZones].zoneShape.shapeType = PrismSquareBase_Z;
        zones.Zones[zones.NbZones].zoneShape.Points[0].X = zone[0];
        zones.Zones[zones.NbZones].zoneShape.Points[0].Y = zone[1];
        zones.Zones[zones.NbZones].zoneShape.Points[0].Z = zone[2];
        zones.Zones[zones.NbZones].zoneShape.Points[0].ThetaX = zone[3];
        zones.Zones[zones.NbZones].zoneShape.Points[0].ThetaY = zone[4];
        zones.Zones[zones.NbZones].zoneShape.Points[0].ThetaZ = zone[5];
        zones.Zones[zones.NbZones].zoneShape.Points[1].X = zone[6];
        zones.Zones[zones.NbZones].zoneShape.Points[1].Y = zone[7];
        zones.Zones[zones.NbZones].zoneShape.Points[1].Z = zone[8];
        zones.Zones[zones.NbZones].zoneShape.Points[2].X = zone[9];
        zones.Zones[zones.NbZones].zoneShape.Points[2].Y = zone[10];
        zones.Zones[zones.NbZones].zoneShape.Points[2].Z = zone[11];
        zones.Zones[zones.NbZones].zoneShape.Points[3].X = zone[12];
        zones.Zones[zones.NbZones].zoneShape.Points[3].Y = zone[13];
        zones.Zones[zones.NbZones].zoneShape.Points[3].Z = zone[14];
        zones.Zones[zones.NbZones].zoneShape.Points[4].Z = zone[15];
        zones.Zones[zones.NbZones].zoneLimitation.speedParameter1 = zone[16];
        zones.Zones[zones.NbZones].zoneLimitation.speedParameter2 = zone[17];

        MySetProtectionZone(zones);
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        return false;
    }
}

bool getGlobalTrajectoryInfo(double *info)
{
    TrajectoryFIFO FIFO;
    if (MyInitAPI != NULL && commandLayer_handle != NULL && MyGetGlobalTrajectoryInfo != NULL)
    {
        MyGetGlobalTrajectoryInfo(FIFO);
        info[0] = FIFO.TrajectoryCount;
        info[1] = FIFO.UsedPercentage;
        info[2] = FIFO.MaxSize;
        return true;
    }
    
    else
    {
        mexPrintf("Librairy or API not open\n");
        
        return false;
    }
}
