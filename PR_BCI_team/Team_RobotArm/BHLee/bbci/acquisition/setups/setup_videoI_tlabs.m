
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath(genpath([BCI_DIR 'acquisition/setups/video_tlabs']));
addpath([BCI_DIR 'acquisition/setups/season10']);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n*********************\nWelcome to video quality \n*********************\n');
%% Start Brainvision recorder
system('c:\BrainVision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'FastnEasy_TLabs_64ch_ActiCapMontageApprox');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Make folder for saving EEG data
acq_makeDataFolder;
global TODAY_DIR

%% Set Screensizes
Screen_Thinkpad = [1440 900];
Screen_External = [1680 1050];
VideoSize = [1280 720];
Offset = (Screen_External-VideoSize)/2;
VP_SCREEN = [Screen_Thinkpad(1) 1 Screen_External];

% Trigger and timing
RUN_START = 254;
RUN_END = 255;
TRIAL_START = 250;
TRIAL_END = 251;
PAUSE_START = 248;
PAUSE_END = 249;

t_wait = 2;   % time in s after each video
N_perVid = 100;  % number of presentations of one video

R_LQ = 'R 11';
R_HQ = 'R  7';
R_LED = 'R 31';
Markers = {R_LQ, R_HQ, R_LED};

% Video settings 
global VLC_DIR
VLC_DIR = '"C:\Program Files (x86)\VLC"';
videodir = 'C:\data\videos\HC3\';      % directory that is scanned for vids

% VLC options
global VLC_OPTS
VLC_OPTS = [' --file-caching=2000 --play-and-exit --no-loop --no-video-title-show --no-video-deco --no-autoscale --no-media-library -I dummy --no-embedded-video --video-x=' int2str(1440+Offset(1)) ' --video-y=' int2str(1+Offset(2)) ' '];
vids = dir(fullfile(videodir,'*.avi'));
vids = struct2cell(vids);
vids = vids(1,:);  % filenames

% Command Line Trial Counter
count = counter();




fprintf('Type ''vormessungen'' and press <RET>\n');



