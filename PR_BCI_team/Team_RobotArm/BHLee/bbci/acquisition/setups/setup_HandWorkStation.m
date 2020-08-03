
addpath([BCI_DIR 'acquisition/setups/HandWorkStation']);
addpath([BCI_DIR 'acquisition/setups/season10']);
fprintf('\n\n********** Welcome to the HandWorkStation **********\n\n');
  
%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'HandworkStation');
try
    bvr_checkparport('type','S');
catch
    error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Check VP_CODE, initialize counter, and create data folder
if isempty(VP_CODE)
    warning('VP_CODE undefined - assuming fresh subject');
end
if strcmp(VP_CODE, 'Temp')
    VP_NUMBER = 1;
else
    VP_COUNTER_FILE = [DATA_DIR 'HandWorkStation_VP_Counter'];
    if exist([VP_COUNTER_FILE '.mat']),
        load(VP_COUNTER_FILE, 'VP_NUMBER');
    else
        VP_NUMBER = 0;
    end
    VP_NUMBER = VP_NUMBER + 1;
    fprintf('VP number %d.\n', VP_NUMBER);
end
global TODAY_DIR
acq_makeDataFolder;

%% 
acqFolder = [BCI_DIR 'acquisition/setups/HandWorkStation/'];
setenv('PATH',['C:\Python26;' getenv('PATH')])

RUN_END = {'S255'};
VP_SCREEN = [0 0 1920 1080];

fprintf('Step 1: Type ''run_HandWorkStation_Practice'' and press <RETURN> to start practicing\n');
fprintf('Step 2: Type ''run_HandWorkStation_Calibration_1'' and press <RETURN> to start speed calibration\n');
fprintf('Step 3: Enter optimal screw speed in ''run_HandWorkStation_Calibration_2.mat''\n');
fprintf('Step 4: Type ''run_HandWorkStation_Calibration_2'' and press <RETURN> to start distance calibration\n');
fprintf('Step 5: Enter optimal screw distance and speed in ''run_HandWorkStation_Game.mat''\n');
fprintf('Step 6: Type ''run_HandWorkStation_ArtifactsRelax'' and press <RETURN> to start artifact and relax measurements\n');
fprintf('Step 7: Type ''run_HandWorkStation_Game'' and press <RETURN> to start main experiment\n');
