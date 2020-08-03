% Slideshow.m
%
% Demonstrates features of iView X API in the matlab toolbox
% This script shows how to set up an SlideShow experiment using iViewX API
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


[pSystemInfoData, pSample32Data, pEvent32Data, pAccuracyData, Calibration] = InitiViewXAPI();

Calibration.method = int32(5);
Calibration.visualization = int32(1);
Calibration.displayDevice = int32(0);
Calibration.speed = int32(0);
Calibration.autoAccept = int32(1);
Calibration.foregroundBrightness = int32(20);
Calibration.backgroundBrightness = int32(239);
Calibration.targetShape = int32(1);
Calibration.targetSize = int32(10);
Calibration.targetFilename = int8('');
pCalibrationData = libpointer('CalibrationStruct', Calibration);


% disp('Define Logger')
% calllib('iViewXAPI', 'iV_SetLogger', int32(1), int8('D:\\iViewXSDK_Matlab_Slideshow_Demo.txt'))


disp('Connect to iViewX')
calllib('iViewXAPI', 'iV_Connect', int8('127.0.0.1'), int32(4444), int8('127.0.0.1'), int32(5555))


% disp('Show Tracking Monitor')
% calllib('iViewXAPI', 'iV_ShowTrackingMonitor');


% disp('Send Command to iViewX')
% calllib('iViewXAPI', 'iV_SendCommand', int8('Hello iView from Matlab'))


disp('Get System Info Data')
calllib('iViewXAPI', 'iV_GetSystemInfo', pSystemInfoData)
get(pSystemInfoData, 'Value')

            
disp('Calibrate iViewX')
calllib('iViewXAPI', 'iV_SetupCalibration', pCalibrationData)
calllib('iViewXAPI', 'iV_Calibrate')


disp('Validate Calibration')
calllib('iViewXAPI', 'iV_Validate')



disp('Show Accuracy')
calllib('iViewXAPI', 'iV_GetAccuracy', pAccuracyData, int32(0))
get(pAccuracyData,'Value')

image1 = imread('1280\image01.jpg');
image2 = imread('1280\image02.jpg');
image3 = imread('1280\image03.jpg');

try 

    % define screen, needs to be full size
    window = Screen('OpenWindow', 0);
    HideCursor;  

    % clear recording buffer
    calllib('iViewXAPI', 'iV_ClearRecordingBuffer');

    % start recording
    calllib('iViewXAPI', 'iV_StartRecording');

    % show first image 
    calllib('iViewXAPI', 'iV_SendImageMessage', int8('image01.jpg'));
    texture1 = Screen('MakeTexture', window, image1);
    Screen('DrawTexture', window, texture1);
    Screen(window, 'Flip');
    pause(3);

    % show second image 
    calllib('iViewXAPI', 'iV_SendImageMessage', int8('image02.jpg'));
    texture2 = Screen('MakeTexture', window, image2);
    Screen('DrawTexture', window, texture2);
    Screen(window, 'Flip');
    pause(3);

    % show third image 
    calllib('iViewXAPI', 'iV_SendImageMessage', int8('image03.jpg'));
    texture3 = Screen('MakeTexture', window, image3);
    Screen('DrawTexture', window, texture3);
    Screen(window, 'Flip');
    pause(3);

    % stop recording
    calllib('iViewXAPI', 'iV_StopRecording');

    % save recorded data
    user = int8('User1');
    description = int8('Description1');
    ovr = int32(1);
    filename = int8(['D:\iViewXSDK_Matlab_Slideshow_Data_' user '.idf']);
    calllib('iViewXAPI', 'iV_SaveData', filename, description, user, ovr)

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


