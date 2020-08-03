% iViewXAPI.m
%
% Initializes iViewX API structures
%
% Author: SMI GmbH
% Feb. 17, 2011

%===========================
%==== Function definition
%===========================

function [pSystemInfoData, pSample32Data, pEvent32Data, pAccuracyData, Calibration] = InitiViewXAPI()


%===========================
%==== System Info
%===========================

SystemInfo.samplerate = int32(0);
SystemInfo.iV_MajorVersion = int32(0);
SystemInfo.iV_MinorVersion = int32(0);
SystemInfo.iV_Buildnumber = int32(0);
SystemInfo.API_MajorVersion = int32(0);
SystemInfo.API_MinorVersion = int32(0);
SystemInfo.API_Buildnumber = int32(0);
SystemInfo.iV_ETDevice = int32(0);
pSystemInfoData = libpointer('SystemInfoStruct', SystemInfo);


%===========================
%==== Eye data
%===========================

Eye.gazeX = int32(0);
Eye.gazeY = int32(0);
Eye.diam = int32(0);
Eye.eyePositionX = int32(0);
Eye.eyePositionY = int32(0);
Eye.eyePositionZ = int32(0);


%===========================
%==== Online Sample data
%===========================

Sample32.timestamp = double(0);
Sample32.leftEye = Eye;
Sample32.rightEye = Eye;
Sample32.planeNumber = int32(0);
pSample32Data = libpointer('SampleStruct32', Sample32);


%===========================
%==== Online Event data
%===========================

Event32.eventType = int8('F');
Event32.eye = int8('l');
Event32.startTime = double(0);
Event32.endTime = double(0);
Event32.duration = double(0);
Event32.positionX = double(0);
Event32.positionY = double(0);
pEvent32Data = libpointer('EventStruct32', Event32);


%===========================
%==== Accuracy data
%===========================

Accuracy.deviationLX = double(0);
Accuracy.deviationLY = double(0);
Accuracy.deviationRX = double(0);
Accuracy.deviationRY = double(0);
pAccuracyData = libpointer('AccuracyStruct', Accuracy);


%===========================
%==== Calibration data
%===========================

Calibration.method = int32(5);
Calibration.visualization = int32(1);
Calibration.displayDevice = int32(0);
Calibration.speed = int32(0);
Calibration.autoAccept = int32(1);
Calibration.foregroundBrightness = int32(20);
Calibration.backgroundBrightness = int32(239);
Calibration.targetShape = int32(1);
Calibration.targetSize = int32(15);
Calibration.targetFilename = int8([0:255] * 0 + 30);





