%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%% Experimental settings
nVP = 14;
VEP_file = [BCI_DIR 'acquisition/setups/labrotation10_Helene/VEP_feedback'];
P300_file = [BCI_DIR 'acquisition/setups/labrotation10_Helene/p300_feedback'];
symmetry_visual_file = [BCI_DIR 'acquisition/setups/labrotation10_Helene/symmetry_visual'];
symmetry_auditory_file = [BCI_DIR 'acquisition/setups/labrotation10_Helene/symmetry_auditory'];
symmetry_passive_file = [BCI_DIR 'acquisition/setups/labrotation10_Helene/symmetry_passive'];
symmetry_basedir = 'D:\\stimuli\\labrotation10_Helene';

% Audio Sachen
vlcdir = 'C:\Program Files\VideoLAN\VLC';
% vlcdir = 'C:\Programme\VideoLAN\VLC';
audiodir = 'D:\stimuli\labrotation10_Helene\audio\';
audiofile = { {'BeautyBeast' 'said'} {'BootsAndHisBrothers' 'they'}  ...
  {'OneTwoThreeEyes' 'said'} {'SleepingBeauty' 'they'}};
videoopt = '';
typ = 'A';   % stimulus typ: A=natural B=blob
COUNTDOWN_START = 'S240';
RUN_END = 'S255';


% Eyetracker settings
et_range = 140; %140
et_range_time = 200;

%% Artifacts
[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
fprintf('Press <RETURN> to start TEST artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt, 'test',1);
[seq, wav, opt]= setup_season10_artifacts('clstag', '');
fprintf('Press <RETURN> to start artifact measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
close all

%% Startup pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks\Symmetry'],'gui',0);
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', '127.0.0.1', 12345);

%% Vormessung: VEP
pyff('init','CheckerboardVEP');pause(.5)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start VEP measurement.\n'); pause;
fprintf('Ok, starting...\n'),close all
% pyff('setdir', '');
pyff('setdir', 'basename', 'VEP');
pause(5)
pyff('play')
pause(5)
stimutil_waitForMarker(RUN_END);
fprintf('VEP measurement finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Vormessung: P300
pyff('init','P300_Rectangle'); pause(.5)
pyff('load_settings', P300_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start P300 measurement.\n'); pause;
fprintf('Ok, starting...\n'),close all
% pyff('setdir', '');
pyff('setdir', 'basename', 'P300');
pause(5)
pyff('play')
pause(5)
stimutil_waitForMarker(RUN_END);
fprintf('P300 measurement finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');
fprintf('Press <RET> to continue.\n'); pause;


%% Symmetry -- practice NOISE
nPracticeTrials = 20;

fprintf('Press <RETURN> to start symmetry NOISE practice.\n'); pause;
pyff('init','Symmetry2'); pause(.5)
pyff('load_settings', symmetry_passive_file);
pyff('set','basedir',symmetry_basedir);
pyff('set','targetPrefix','symA','nontargetPrefix','ranA');
pyff('set','et_range',et_range,'et_range_time',et_range_time);
pyff('setint','screenPos',VP_SCREEN);
pyff('setint','nTrials',nPracticeTrials);
%pyff('setint','use_eyetracker',0);
pyff('setdir', '');
pause(5)
fprintf('Ok, starting...\n'), close all
pyff('play')
pause(5)
stimutil_waitForMarker(RUN_END);
fprintf('Symmetry practice finished!\n')
% bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');

%% Symmetry -- practice BLOBS
nPracticeTrials = 15;

fprintf('Press <RETURN> to start symmetry BLOBS practice.\n'); pause;
pyff('init','Symmetry2'); pause(.5)
pyff('load_settings', symmetry_passive_file);
pyff('set','basedir',symmetry_basedir);
pyff('set','targetPrefix','symB','nontargetPrefix','ranB');
pyff('set','et_range',et_range,'et_range_time',et_range_time);
pyff('setint','screenPos',VP_SCREEN);
pyff('setint','nTrials',nPracticeTrials);
% pyff('setint','use_eyetracker',0);
pyff('setdir', '');
pause(5)
fprintf('Ok, starting...\n'), close all
pyff('play')
pause(30)
stimutil_waitForMarker(RUN_END);
fprintf('Symmetry practice finished!\n')
% bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');

%% *** Symmetry RUNS ***

%% Symmetry PASSIVE
order = [mod(nVP,2) 1-mod(nVP,2)]+1;
orderType = {'natural' 'blob'};
orderPrefix = {'A' 'B'};
nRepetitions = 2;
nTrials = 464;

for ii=1:nRepetitions
  for oo=order
    fprintf('Press <RETURN> to start passive symmetry block #%d (%s) run.\n',ii,orderType{oo}); pause;
    pyff('init','Symmetry2'); pause(.5)
    pyff('load_settings', symmetry_passive_file);
    pyff('set','targetPrefix',['sym' orderPrefix{oo}],'nontargetPrefix',['ran' orderPrefix{oo}]);
    pyff('setdir', 'basename', ['symmetry_passive_' orderType{oo}]);
    pyff('set','et_range',et_range,'et_range_time',et_range_time);
    pyff('setint','screenPos',VP_SCREEN);
    pyff('setint','nTrials',nTrials/nRepetitions);
    pause(5)
    fprintf('Ok, starting...\n'), close all
    pyff('play')
    pause(30)
    stimutil_waitForMarker(RUN_END);
    fprintf('Symmetry passive run finished!\n')
    bvr_sendcommand('stoprecording');
    pyff('stop');
    pyff('quit');
  end
end

%% Symmetry AUDITORY and VISUAL
order = [mod(floor(nVP/2),2) 1-mod(floor(nVP/2),2)]+1; % 1->2->2->1
nRepetitions = 2;
audiofile = audiofile(randperm(4));   % Bring audiofiles in random order
audionum = 1;

for ii=1:nRepetitions
  for oo=order
    fprintf('Press <RETURN> to start symmetry block #%d run.\n',ii); pause;
    pyff('init','Symmetry2'); pause(.5)
    if oo==1
      pyff('load_settings', symmetry_auditory_file);
      pyff('setdir','basename','symmetry_auditory');
      pyff('set','word',audiofile{audionum}{2});
    else
      pyff('load_settings', symmetry_visual_file);
      pyff('setdir','basename','symmetry_visual');
      pyff('set','word','Please count the stimulus repetitions!');
    end
    pyff('setint','screenPos',VP_SCREEN);
    pyff('setint','fullscreen',0);
    pyff('set','targetPrefix',['sym' typ],'nontargetPrefix',['ran' typ]);
    pause(.5)
    fprintf('Ok, starting...\n'), close all
    pyff('play')
    stimutil_waitForMarker(COUNTDOWN_START);
    fprintf('Detected RUN_START\n')
    pause(3)
    cmd= ['cmd /C "C: & cd ' vlcdir ' & vlc ' videoopt ' -vvv  ' [audiodir audiofile{audionum}{1} '.mp3'] ' vlc://quit"']; % call vlc & quit after playing
    system(cmd)
    pause(30)
    stimutil_waitForMarker(RUN_END);
    fprintf('Symmetry run finished!\n')
    bvr_sendcommand('stoprecording');
    pyff('stop');
    pyff('quit');
    audionum = audionum+1;
  end
end

%%
fprintf('Experiment finished.\n');