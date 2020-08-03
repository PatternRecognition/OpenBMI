
%===========================
% Example program
%===========================
%
% For further information, refer to the iViewX and iViewX SDK manuals.
%

global DATA_DIR
EYE_DIR = [DATA_DIR 'eyeRaw' filesep];

% ethernet connection settings (cf. iViewX 2 manual, Section 6.2.)
opt.ip_smi = '169.254.140.36';  % IP address of (remote) SMI laptop
%opt.ip_bbci = '169.254.247.97'; %252.149';  % IP address of (this) BBCI laptop
opt.ip_bbci = '130.149.83.92'; % IP address of BBCILAPTOP07
opt.port_smi = 4444;

et =  smi('init')
out_flag = smi('connect',opt)
out_flag = smi('test_connection')
keyboard
out_flag = smi('iV_ShowTrackingMonitor');

% Calibrate & Validate            
disp('Calibrate iViewX')
smi('SetupCalibration', et.pCalibrationData)
smi('iV_Calibrate')

disp('Validate Calibration')
smi('iV_Validate')

disp('Show Accuracy')
smi('iV_GetAccuracy', et.pAccuracyData, int32(0))
get(et.pAccuracyData,'Value')


% Start recording
try 

    % clear recording buffer
    smi('iV_ClearRecordingBuffer');

    % start recording
    smi('iV_StartRecording');
    pause(5)
    
    % stop recording
    smi('iV_StopRecording');

    % save recorded data
    sbj = int8('_TEST');
    description = int8('');
    ovr = int32(1);
    filename = int8([EYE_DIR' sbj '.idf']);
    smi('iV_SaveData', filename, description, sbj, ovr)

catch e
    disp(e)
end

% disconnect from iViewX 
smi('iV_Disconnect')
pause(1);
clear all
% unload iViewX API library
smi('outit');


