workingdir= [BCI_DIR 'investigation/projects/HandWorkStation/online_detector'];
addpath(workingdir);

%% Start BrainVisionn Recorder, load workspace and check triggers
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', 'HandworkStation_SmA');
try
  bvr_checkparport;
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end


%% Check VP_CODE, initialize counter, and create data folder
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject'); %#ok<WNTAG>
end
global TODAY_DIR
acq_makeDataFolder('multiple_folders', 1);

global VP_SCREEN
VP_SCREEN= [1 31 1366 719];

%% RELAX
addpath([BCI_DIR 'acquisition/setups/season10']);
[seq, wav, opt] = setup_season10_relax;
stimutil_waitForInput('msg_next','to practice RELAX.');
seq_test= strrep(seq, 'R[10]', 'R[3]');
stim_artifactMeasurement(seq_test, wav, opt, 'test',1);
stimutil_waitForInput('msg_next','to start RELAX.');
stim_artifactMeasurement(seq, wav, opt);


%% Eye Movements
[seq, wav, opt] = setup_season10_artifacts_demo('clstag', '');
stimutil_waitForInput('msg_next','to TEST eye movement calibration');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt] = setup_season10_artifacts('clstag', '');
stimutil_waitForInput('msg_next','to START eye movement calibration');
stim_artifactMeasurement(seq, wav, opt);


%% Initialize communication by UDP
% IP of SmA PC: 192.168.1.23
send_xmlcmd_udp('init', '192.168.1.23', 12345);


%% Initialize Workload Detector
wld = HandWorkStation_Initialize;
wld.speed.initial = wld.speed.calibration(1);


%% CALIBRATION 1 (offline)
stimutil_waitForInput('msg_next','to start CALIBRATION  1');

speed = 0;
file_name= ['calibration1_HandWorkStation' VP_CODE];
bvr_startrecording(file_name); pause(1);
for k = 1:wld.calibration.nBlocks,
  speed = 1 + mod(speed, 2);
  ppTrigger(speed*10);
  send_xmlcmd_udp(wld.control_str, wld.speed.calibration(speed));
  pause(wld.calibration.ISI);
  ppTrigger(speed*10+1);
  pause(.1)
end
ppTrigger(255); pause(.5);
bvr_sendcommand('stoprecording');
send_xmlcmd_udp('i:bbci_act_output', 0);


%% CHECK MOVEMENT ARTIFACTS
stimutil_waitForInput('msg_next', 'to check movement artifacts');
files = {[TODAY_DIR file_name]};
online_test_artifacts(wld,files{1});


%% TRAIN CLASSIFIER 1
stimutil_waitForInput('msg_next', 'to train the prelim classifier');
wld = online_train_classifier(wld,files);


%% INITIALIZE ONLINE SYSTEM
clear bbci_control_HandWorkStation
wld.control = false;
wld = online_initialize(wld);
bbci = HandWorkStation_Setup_BBCI(wld);


%% CALIBRATION 2 (online)
stimutil_waitForInput('msg_next','to start CALIBRATION  2');

cmd = '';
cmd = strcat(cmd, ['ISI_list= [' sprintf('%.0f ', wld.calibration.ISI_list) '];']);
cmd = strcat(cmd, ['speed_level= [' sprintf('%.0f ', wld.speed.calibration) '];']);
cmd = strcat(cmd, ['cd ' workingdir '; dbstop if error; HandWorkStation_Calibration2']);

bvr_startrecording(strrep(file_name,'1','2'));
system(['matlab -nosplash -r "' cmd '; exit &']); 
bbci_apply(bbci);
pause(2);
bvr_sendcommand('stoprecording');
send_xmlcmd_udp('i:bbci_act_output', 0);

saveas(wld.fig.h,[TODAY_DIR 'Calibration.fig']);


%% CHECK MOVEMENT ARTIFACTS
stimutil_waitForInput('msg_next', 'to check movement artifacts');
files{2} = strrep(files{1},'on1','on2');
online_test_artifacts(wld,files{2});


%% TRAIN CLASSIFIER 2
stimutil_waitForInput('msg_next', 'to train the final classifier');
wld = online_train_classifier(wld,files);


%% SMA Online Control 1
stimutil_waitForInput('msg_next','to start SMA ONLINE CONTROL 1');

clear bbci_control_HandWorkStation
wld = online_initialize(wld);
wld.speed.initial = wld.speed.calibration(2);
wld.control = true;
bbci = HandWorkStation_Setup_BBCI(wld);

bvr_startrecording(['online_HandWorkStation' VP_CODE]);
bbci_apply(bbci);
bvr_sendcommand('stoprecording');
send_xmlcmd_udp('i:bbci_act_output', 0);

saveas(wld.fig.h,[TODAY_DIR 'Online1.fig']);


%% SMA Online Control 2
stimutil_waitForInput('msg_next','to start SMA ONLINE CONTROL 2');

clear bbci_control_HandWorkStation
wld = online_initialize(wld);
wld.speed.initial = wld.speed.calibration(2);
wld.control = true;
bbci = HandWorkStation_Setup_BBCI(wld);

bvr_startrecording(['online_HandWorkStation' VP_CODE]);
bbci_apply(bbci);
bvr_sendcommand('stoprecording');
send_xmlcmd_udp('i:bbci_act_output', 0);

saveas(wld.fig.h,[TODAY_DIR 'Online3.fig']);

