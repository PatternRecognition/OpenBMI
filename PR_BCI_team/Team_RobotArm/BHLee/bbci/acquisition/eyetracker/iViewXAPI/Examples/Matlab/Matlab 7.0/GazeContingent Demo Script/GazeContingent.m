% GazeContingent.m
%
% This script shows how to set up an Gaze Contingent experiment in Matlab
%
% Author: SMI GmbH
% Jun. 30, 2011

%===========================
%Initialisation
%===========================

%clear all variables, connections, ...
clear all
clc

% load the iViewX API library
loadlibrary('iViewXAPI.dll', 'iViewXAPI.h');


[pSystemInfoData, pSample32, pEvent32, pAccuracy, CalibrationData] = InitiViewXAPI()

CalibrationData.method = int32(5);
CalibrationData.visualization = int32(1);
CalibrationData.displayDevice = int32(0);
CalibrationData.speed = int32(0);
CalibrationData.autoAccept = int32(1);
CalibrationData.foregroundBrightness = int32(20);
CalibrationData.backgroundBrightness = int32(239);
CalibrationData.targetShape = int32(1);
CalibrationData.targetSize = int32(10);
pCalibrationData = libpointer('CalibrationStruct', CalibrationData);


% disp('Define Logger')
% calllib('iViewXAPI', 'iV_SetLogger', int32(1), 'D:\\iViewXSDK_Matlab_GazeContingent_Demo.txt')


disp('Connect to iViewX')
calllib('iViewXAPI', 'iV_Connect', '127.0.0.1', int32(4444), '127.0.0.1', int32(5555))


% disp('Show Tracking Monitor')
% calllib('iViewXAPI', 'iV_ShowTrackingMonitor');


% disp('Send Command to iViewX')
% calllib('iViewXAPI', 'iV_SendCommand', 'Hello iView from Matlab')


disp('Get System Info Data')
calllib('iViewXAPI', 'iV_GetSystemInfo', pSystemInfoData)
get(pSystemInfoData, 'Value')


disp('Calibrate iViewX')
calllib('iViewXAPI', 'iV_SetupCalibration', pCalibrationData)
calllib('iViewXAPI', 'iV_Calibrate')


disp('Validate Calibration')
calllib('iViewXAPI', 'iV_Validate')


disp('Show Accuracy')
calllib('iViewXAPI', 'iV_GetAccuracy', pAccuracy, int32(0))
get(pAccuracy,'Value')


exitLoop = 0;
try 

    % define screen, needs to be adjusted to different stimulus screens
    window = Screen('OpenWindow', 0);
    HideCursor;  
    
    while ~(exitLoop)         

        if (calllib('iViewXAPI', 'iV_GetSample32', pSample32) == 1)

            % get sample
            get(pSample32, 'Value');
            Smp = libstruct('SampleStruct32', pSample32);

            x0 = Smp.leftEye.gazeX;
            y0 = Smp.leftEye.gazeY;
            Screen('DrawDots', window, [x0,y0], 10, [0 0 0])
            Screen(window, 'Flip')
            WaitSecs(0.001)

        end        
        
        % end experiment after a mouse button has been pushed
        if (waitForMouseNonBlocking)
            exitLoop = 1;
        end
    end

catch
    % catch if errors appears
    Screen('CloseAll'); 
    ShowCursor
    s = lasterror
end

disp('Disconnect')

% release screen
Screen('CloseAll'); 
ShowCursor

% disconnect from iViewX 
calllib('iViewXAPI', 'iV_Disconnect')

pause(1);
clear all

% unload iViewX API libraray
unloadlibrary('iViewXAPI');


