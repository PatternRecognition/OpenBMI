acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
CENTERSPELLER_file = [acqFolder 'CenterSpeller_feedback'];
ODDBALL_file = [acqFolder 'Oddball_feedback'];

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);


VERY_SHORT = 0.1;
SHORT      = 0.5;
LONG       = 1.0;
VERY_LONG  = 1.5;
BLOCK_DUR  = 360; % fixed block duration of 6 minutes
STIM_DUR   = 0.1;
delta = 1/60*0.4;

%
%% ----- Oddball - 4 different ISIs, randomized order
%
% order = {[SHORT, LONG, VERY_LONG, VERY_SHORT, VERY_SHORT, ...
%           SHORT, LONG, VERY_LONG, VERY_SHORT, VERY_SHORT]};
possible_orders = {[SHORT, LONG, VERY_LONG, VERY_SHORT, VERY_LONG]
                   [VERY_LONG, LONG, SHORT, VERY_LONG, VERY_SHORT]
                   [LONG, VERY_LONG, SHORT, VERY_SHORT, VERY_LONG]
                   [SHORT, VERY_LONG, LONG, VERY_LONG, VERY_SHORT]
                   [VERY_LONG, LONG, SHORT, VERY_LONG, VERY_SHORT]
                   [SHORT, VERY_SHORT, VERY_LONG, LONG, VERY_LONG]};

selected_order = possible_orders{1+mod(VP_NUMBER-1,length(possible_orders))};

%% Prepare the EEG cap
bvr_sendcommand('checkimpedances');
stimutil_waitForInput('msg_next','when preparation of the cap is finished.');
bvr_sendcommand('viewsignals');


%% Oddball - Practice
ok= 0;
while ~ok,
  pyff('init', 'VisualOddballVE'); 
  pause(4.0);
  pyff('load_settings', ODDBALL_file);
  
  pause(4.0);
  %pyff('setint', 'geometry',VP_SCREEN);
  pyff('setint', 'nTrials',10);
  pyff('set', 'prestim_ival', VERY_LONG-delta, 'stim_duration',STIM_DUR-delta);
  
  pyff('save_settings', 'd:\data\tmp\odd');
   pause(4.0);
  odd= pyff_loadSettings('d:\data\tmp\odd.json');
  ok= odd.fullscreen==1;
	if ~ok,
		fprintf('settings not corrected received - retrying\n');
    pyff('quit');
	end
end
pause(4.0);
stimutil_waitForInput('msg_next','to start Oddball practice.');

pyff('play');
stimutil_waitForMarker (RUN_END);
fprintf('Oddball practice finished.\n')
pyff('quit');

%% Oddball - Recording
for i=1:length(selected_order)
  ISI = selected_order(i);
  nTrials = 2*round(BLOCK_DUR/(ISI+STIM_DUR)/2);
  ISIstr= sprintf('%dms', round(1000*ISI)); 
  pyff('init', 'VisualOddballVE'); pause(.5);
  pause(6.0);
  pyff('load_settings', ODDBALL_file);
  pause(6.0);
  %pyff('setint','geometry',VP_SCREEN);
  pyff('setint', 'nTrials', nTrials);
  pyff('setint', 'nTrials_per_block', round(nTrials/2));
  pyff('set', 'prestim_ival', ISI-delta, 'stim_duration',STIM_DUR-delta);
  msg= sprintf('to start Oddball with ISI=%s, nTrials=%d', ISIstr, nTrials);
  stimutil_waitForInput('msg_next',msg);
  pyff('play', 'basename', ['Oddball_ISI' ISIstr], 'impedances', 0);
  stimutil_waitForMarker(RUN_END);
  fprintf('Oddball recording finished.\n')
  pyff('quit');
end



%
%% ----- Center Speller ("VERY_SHORT" ISI) - calibration & free-spelling
%

phrase_practice= 'BCI';
phrase_calibration= 'XYLOPHONSPIELER';
speller= 'CenterSpeller';
msg= ['to start ' speller ' '];


%% Practice
stimutil_waitForInput('msg_next', [msg 'practice']);
setup_speller
pyff('set', 'desired_phrase',phrase_practice)
pyff('setint', 'offline',1);
pyff('play');
stimutil_waitForMarker(RUN_END, 'verbose',1);
pyff('quit');

%% Calibration
stimutil_waitForInput('msg_next', [msg 'calibration']);
setup_speller
pyff('set', 'desired_phrase',phrase_calibration)
pyff('setint', 'offline',1);
pyff('setint', 'nr_sequences', 10);
pyff('play', 'basename', ['calibration_' speller], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');    

%% Train the classifier
feedback_settings= pyff_loadSettings(CENTERSPELLER_file);
bbci.calibrate.file= ['calibration_' speller '*'];
bbci.calibrate.save.file= ['bbci_classifier_' speller VP_CODE];
bbci.calibrate.settings.nSequences= 5;

[bbci, data]= bbci_calibrate(bbci);
bbci= copy_subfields(bbci, bbci_default);
bbci_save(bbci, data);

%% Freespelling
stimutil_waitForInput('msg_next', [msg 'free-spelling']);
setup_speller
pyff('set', 'desired_phrase','');
pyff('setint', 'offline',0);
pyff('setint', 'nr_sequences', 5);
pyff('set', 'copy_spelling', false);
pyff('play', 'basename', ['free_' speller], 'impedances', 0);
pause(1)
bbci_apply(bbci);
%% To stop the recording: type 'ppTrigger(255)' in a second Matlab
fprintf('Free-spelling run finished.\n')
pyff('quit');



%
%% ----- Center Speller ("VERY_LONG" ISI) - calibration only
%

%% Calibration - Part 1
stimutil_waitForInput('msg_next', [msg 'SLOW calibration']);
setup_speller
pyff('set', 'desired_phrase','FRAU')
pyff('setint', 'offline',1);
pyff('setint', 'nr_sequences', 5);
pyff('set', 'interstimulus_duration', VERY_LONG - 1/60/2);
pyff('play', 'basename', ['calibration_Slow_' speller], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');    


%% Calibration - Part 2
stimutil_waitForInput('msg_next', [msg 'SLOW calibration']);
setup_speller
pyff('set', 'desired_phrase','KIND')
pyff('setint', 'offline',1);
pyff('setint', 'nr_sequences', 5);
pyff('set', 'interstimulus_duration', VERY_LONG - 1/60/2);
pyff('play', 'basename', ['calibration_Slow_' speller], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');    



%% --- - --- Session finished

acq_vpcounter(session_name, 'close');
fprintf('Session %s finished.\n', session_name);
