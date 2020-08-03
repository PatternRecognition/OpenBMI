
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

TEST = 1;

addpath([BCI_DIR 'acquisition/setups/videoI_hhi']);
addpath([BCI_DIR 'acquisition/setups/season10']);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n*********************\nWelcome to video quality \n*********************\n');

%% Start Brainvision recorder
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');

if ~TEST
  try
    bvr_checkparport('type','S');
  catch
    error('Check amplifiers (all switched on?) and trigger cables.');
  end
end

%% Make folder for saving EEG data
acq_makeDataFolder('multiple_folders',1);
global TODAY_DIR

% Trigger and timing
RUN_START = 254;
RUN_END = 255;
TRIAL_START = 250;
TRIAL_END = 251;
PAUSE_START = 248;
PAUSE_END = 249;

t_wait = 0;   % time in s after each video
N_perVid = 100;  % number of presentations of one video

R_LQ = 'R  4';
R_HQ = 'R  8';
R_LED = 'R  1';
Markers = {R_LQ, R_HQ, R_LED};

% Video settings 
global VLC_DIR
VLC_DIR = '"C:\Program Files\VideoLAN\VLC"';
videodir = 'D:\data\hhi_videos\'; %'F:\clips\';   % directory that is scanned for vids
prefix = 'bigTexture_832x480_60';
cd(VLC_DIR(2:end-1))

% VLC options
global VLC_OPTS
VLC_OPTS = [' --file-caching=2500 --play-and-exit --no-loop --no-video-title-show --scale 1 --no-video-deco --no-autoscale --no-media-library -I dummy --no-embedded-video --width=2560 --height=1600 '];
vids = dir([videodir prefix '*.avi']);
vids = struct2cell(vids);
vids = vids(1,:);  % filenames

cmd_vlc = ['cmd /C "C: & vlc ' VLC_OPTS ' '];

% remove unused LQ conditions
LQs_codec = [2 3 4 5 6 7 8 9 10];
d = []; vidsLQ = cell(1,length(LQs_codec));
for v = 2:length(vids)
  lq = get_ql(vids{v});
  if any(lq==LQs_codec)
     vidsLQ{find(lq==LQs_codec)}{end+1} = vids{v}; 
  end
end
vids = {repmat(vids(1),1,length(vidsLQ{1})),vidsLQ{:}};

% remove the 51th video 
for n = 1:length(vids)
  vids{n} = vids{n}(1:50);
end

% get waffle videos
use_waffle_vids = 1;
prefix = 'bigWaffle_832x480_30';
waffle_vids = dir([videodir prefix '*.avi']);
waffle_vids = struct2cell(waffle_vids);
waffle_vids = waffle_vids(1,:);  % filenames


% Command Line Trial Counter
count = counter();


fprintf('Type ''vormessungen'' and press <RET>\n');
