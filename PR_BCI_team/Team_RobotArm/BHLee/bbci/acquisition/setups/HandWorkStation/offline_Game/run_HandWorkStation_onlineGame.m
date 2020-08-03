

addpath([BCI_DIR 'acquisition/setups/HandWorkStation']);
addpath([BCI_DIR 'investigation/projects/HandWorkStation/online_detector']);
fprintf('\n\n********** Welcome to the HandWorkStation ONLINE Game, Manfred **********\n\n');


%% Start BVR, load workspace, check triggers & init VP_CODE
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'HandworkStation');
try
    bvr_checkparport('type','S');
catch me
    error('Check amplifiers (all switched on?) and trigger cables.');
end

VP_CODE = 'gaa';

global TODAY_DIR
acq_makeDataFolder;


%% INITIALIZE
send_xmlcmd_udp('init', '127.0.0.1', 12345);
acqFolder = [BCI_DIR 'acquisition/setups/HandWorkStation/'];
setenv('PATH',['C:\Python26;' getenv('PATH')])
RUN_END = {'S255'};
VP_SCREEN = [0 0 1920 1080];


%% CALIBRATION 1 (2 runs)
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
for ii = 1:2    
    pyff('init','HandWorkStation2'); pause(1.5)
    pyff('setint','MODE',1);
    pyff('setint','screen_pos',VP_SCREEN);
    
    fprintf('Press <RETURN> to start CALIBRATION %d...\n',ii); pause;
    fprintf('Ok, starting...\n'), close all
    
    pyff('play', 'basename', 'HandWorkStation_Calibration', 'impedances',0);
    stimutil_waitForMarker(RUN_END);
    
    fprintf('HandWorkStationGame Calibration %d finished.\n',ii)
    pyff('quit'); pause(1);
end


%% TRAINING 1 & INIT BBCI
wld = HandWorkStation_Initialize;
wld = online_initialize(wld);

files{1} = [TODAY_DIR 'HandWorkStation_CalibrationVP' VP_CODE];
files{2} = [TODAY_DIR 'HandWorkStation_CalibrationVP' VP_CODE '02'];
wld = online_train_classifier(wld,files);

bbci = HandWorkStation_Setup_BBCI(wld);


%% CALIBRATION 2 (online)
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
pyff('init','HandWorkStation2'); pause(1.5)

pyff('setint','MODE',2);
pyff('setint','screen_pos',VP_SCREEN);

fprintf('Press <RETURN> to start CALIBRATION 3...\n'); pause;
fprintf('Ok, starting...\n'), close all

pyff('play', 'basename', 'HandWorkStation_Calibration', 'impedances',0);
pause(3);
bbci_apply(bbci);
stimutil_waitForMarker(RUN_END);

fprintf('HandWorkStationGame Calibration 3 finished.\n')
pyff('quit'); pause(1);


%% TRAINING 2
files{3} = [TODAY_DIR 'HandWorkStation_CalibrationVP' VP_CODE '03'];
wld = train_Classifier(wld,files);
bbci = HandWorkStation_Setup_BBCI(wld);


%% ONLINE GAME (2 runs)
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);

for ii = 1:2
    
    clear bbci_control_HandWorkStation
    wld = online_initialize(wld);
    wld.control = 1;    % pass speed control to the classifier
    
    pyff('init','HandWorkStation2'); pause(1.5)
    pyff('setint','MODE',3);
    pyff('setint','screen_pos',VP_SCREEN);
    
    fprintf('Press <RETURN> to start ONLINE GAME %d...\n',ii); pause;
    fprintf('Ok, starting...\n'), close all
    
    pyff('play', 'basename', 'HandWorkStation_Online', 'impedances',0);
    pause(3);
    bbci_apply(bbci);
    stimutil_waitForMarker(RUN_END);
    
    fprintf('HandWorkStationGame OnlineGame %d finished.\n',ii)
    pyff('quit'); pause(1);
    
end









