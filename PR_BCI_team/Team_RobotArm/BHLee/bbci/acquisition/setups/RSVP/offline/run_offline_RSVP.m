%% Preparation of the EEG cap
% bvr_sendcommand('checkimpedances');
stimutil_waitForInput('msg_next','when finished preparing the cap.');
bvr_sendcommand('viewsignals');
pause(5);

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

%% Artifacts
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% ** Startup pyff **
close all
pyff('startup','a','D:\svn\bbci\python\pyff\src\Feedbacks')
% system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ... 
%   ' -a D:\svn\bbci\python\pyff\src\Feedbacks  --nogui -l debug -p brainvisionrecorderplugin" &']);
bvr_sendcommand('viewsignals');
pause(8)
% send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
% send_xmlcmd_udp('init', '127.0.0.1', 12345);

%% Standard oddball PRACTICE 
fprintf('Press <RETURN> to start oddball PRACTICE.\n');
pause
setup_oddball
pyff('setint','nStim',20);
pyff('setdir','');
% send_xmlcmd_udp('interaction-signal', 'i:nStim',20);
% send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', ...
%   's:BASENAME','');
pause(.01)
fprintf('Ok, starting...\n'),close all
pyff('play');
% send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
pause(5)
stimutil_waitForMarker('stopmarkers','S253');
fprintf('Practice finished? If yes, press <RETURN>\n'),pause
pyff('stop');
pyff('quit');
% send_xmlcmd_udp('interaction-signal', 'command', 'stop');pause(1);
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');

%% Standard oddball measurement 
fprintf('Press <RETURN> to start oddball measurement.\n');
pause
setup_oddball
pyff('setdir','basename','oddball');
%   send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, ...
%     's:BASENAME','oddball');
pause(.01)
fprintf('Ok, starting...\n'),close all
pyff('play');
% send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
pause(5)
stimutil_waitForMarker('stopmarkers','S253');
fprintf('Measurement finished? If yes, press <RETURN>\n'),pause
pyff('stop');
pyff('quit');
% send_xmlcmd_udp('interaction-signal', 'command', 'stop');pause(1);
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');
bvr_sendcommand('stoprecording');

%% Experimental settings
alternating_colors = [0 1];   % Color off, Color on
alternating_colors_name = {'Monochrome','Color'};
% stim_durations = [.08333333333 .133333333];  % in s 
stim_durations_name = {'83ms','133ms'};
%stim_durations = [.070 .120];  % in s 
stim_durations = [.09 0.1];  % in s 

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

%% Practice black-white
fprintf('Press <RETURN> to start black-white practice\n'),
close all
pause
current_condition = [1 2 1];
setup_RSVP_feedback
pyff('setint','custom_pre_sequences',practice_pre_sequences);
pyff('setint','custom_post_sequences',practice_post_sequences);
% pyff('set','words',{'ALigE'});
pyff('set','words',{'ALIGE'});
pyff('setdir','');
% send_xmlcmd_udp('interaction-signal', 'i:custom_pre_sequences',practice_pre_sequences);
% send_xmlcmd_udp('interaction-signal', 'i:custom_post_sequences',practice_post_sequences);
% send_xmlcmd_udp('interaction-signal', 's:words', {'WAR'});
% send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
pause(.01)
fprintf('Ok, starting...\n'),close all
pyff('play');
% send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
pause(5)
stimutil_waitForMarker('stopmarkers','S253');
fprintf('Practice finished? If yes, press <RETURN>\n'),pause
pyff('stop');
pyff('quit');
% send_xmlcmd_udp('interaction-signal', 'command', 'stop');pause(1);
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');

%% Practice color
% fprintf('Press <RETURN> to start color practice\n'),pause
current_condition = [2 2 2];
setup_RSVP_feedback
pyff('setint','custom_pre_sequences',practice_pre_sequences);
pyff('setint','custom_post_sequences',practice_post_sequences);
% pyff('set','words',{'TAMOR'});
% pyff('set','words',{'alige'});
% fb.color_groups = {'fdygk-', 'PJUX&lt;E', 'iSwcz/','TBmqAH','LrvOn.'};%test
pyff('set','words',{'ALIGE'});
pyff('setdir','');
% send_xmlcmd_udp('interaction-signal', 'i:custom_pre_sequences',practice_pre_sequences);
% send_xmlcmd_udp('interaction-signal', 'i:custom_post_sequences',practice_post_sequences);
% send_xmlcmd_udp('interaction-signal', 's:words', {'TAMOR'});
% send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
pause(.01)
fprintf('Ok, starting...\n'),close all
pyff('play');

%%
% send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
% pause(5)
% stimutil_waitForMarker('stopmarkers','S253');
% fprintf('Practice finished? If yes, press <RETURN>\n'),pause
pyff('stop');
pyff('quit');
% send_xmlcmd_udp('interaction-signal', 'command', 'stop');pause(1);
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%% Experiment
rand_idx = randperm(length(conditions));
fprintf('Practice finished. Press <RETURN> to start the experiment.\n'),pause

for current_idx = 1:length(rand_idx),
%%
  current_condition = conditions{rand_idx(current_idx)};
  letters_group=words{current_condition(3)};
  group_length=length([letters_group{:}]);
  fprintf('Colormode: %s; Timing: %s\n', alternating_colors_name{current_condition(1)}, stim_durations_name{current_condition(2)});
  fprintf('Next block: #%d.\n',current_idx);
  
  % Prepare feedback
  setup_RSVP_feedback;  
  
  pyff('setint','custom_pre_sequences',custom_pre_sequences);
  pyff('setint','custom_post_sequences',custom_post_sequences);
%   send_xmlcmd_udp('interaction-signal', 'i:custom_pre_sequences',custom_pre_sequences);
%   send_xmlcmd_udp('interaction-signal', 'i:custom_post_sequences',custom_post_sequences);
  
  pyff('setdir','basename',[filename alternating_colors_name{current_condition(1)} stim_durations_name{current_condition(2)}]);
%   send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, ...
%     's:BASENAME',[filename alternating_colors_name{current_condition(1)} stim_durations_name{current_condition(2)}]);
  pause(1)
  stimutil_waitForInput('msg_next','when ready to proceed with feedback.\n');
  % Start
  pyff('play');
%   send_xmlcmd_udp('interaction-signal', 'command', 'play');
  pause(60);  
  fprintf('If the block %i is finished, press <RETURN>!\n',current_idx),pause
  pause(2);
  pyff('stop');
  pyff('quit');
%   send_xmlcmd_udp('interaction-signal', 'command', 'stop');pause(1);
%   send_xmlcmd_udp('interaction-signal', 'command', 'quit');
  bvr_sendcommand('stoprecording');
  current_idx = current_idx+1;
%%  
  
  
end
fprintf('Experiment finished!\n');
