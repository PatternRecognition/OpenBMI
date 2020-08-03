addpath([BCI_DIR 'acquisition/setups/RSVP']);
addpath([BCI_DIR 'acquisition/setups/season10']);

VP_CODE= 'Temp';
global TODAY_DIR
TODAY_DIR= [];
VP_SCREEN = [0 0 1920 1200];

system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ... 
  ' -a D:\svn\bbci\python\pyff\src\Feedbacks  --nogui -l debug -p brainvisionrecorderplugin" &']);

pause(8)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

alternating_colors = [0 1];   % Color off, Color on
alternating_colors_name = {'Monochrome','Color'};
% stim_durations = [.08333333333 .133333333];  % in s 
stim_durations_name = {'83ms','133ms'};
%stim_durations = [.070 .120];  % in s 
stim_durations = [.04 .1];  % in s 


% words =  {{'WINKT','FJORD','LUXUS'} ... 
%           {'SPHINX','QUARZ','VODKA'} ...
%           {'YACHT','GEBOT','MEMME'}};
words =  {{'WINCK','FJORD','LEGUS'} ... 
          {'SPHINX','QUARZ','VAMBT'}};

% Experimental subconditions: [nocolor-color stim_duration block_nr]
% conditions = {[1 1 1] [1 1 2] [1 1 3 ] [1 2 1] [1 2 2 ] [1 2 3] ...
%   [2 1 1] [2 1 2] [2 1 3] [2 2 1] [2 2 2] [2 2 3]};
conditions = {[1 1 1] [1 1 2] [1 2 1] [1 2 2 ]  ...
  [2 1 1] [2 1 2] [2 2 1] [2 2 2]};

%% Basic parameters
filename = 'RSVP_';
minTargets = 0;       % Minimum number of targets for pre/post sequence
maxTargets = 2;       % Maximum number of targets for pre/post sequence
custom_pre_sequences = {[minTargets maxTargets]};
custom_post_sequences = {[minTargets maxTargets]};
practice_pre_sequences = {};
practice_post_sequences = {};

% Reset random-generator seed to produce new random numbers
% Take cputime in ms as basis
rand('seed',cputime*1000)

%% Practice 
current_condition = [2 1 2];

setup_RSVP_feedback
send_xmlcmd_udp('interaction-signal', 'i:custom_pre_sequences',practice_pre_sequences);
send_xmlcmd_udp('interaction-signal', 'i:custom_post_sequences',practice_post_sequences);
send_xmlcmd_udp('interaction-signal', 's:words', {'WAR'});
pause(.01)
fprintf('Ok, starting...\n'),close all
send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
