%% Init: Set the bbci Vars, VP_CODE, Dirs
ppACQ_PREFIX_LETTER= '';
ACQ_LETTER_START= 'f';

VP_CODE='VPfbc'; % Code Reinhard Fissler
if isempty(VP_CODE),
    warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups'], path);

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'reducerbox_64mcc_noEOG');
bvr_sendcommand('loadworkspace', 'reducerbox_64mcc_noEOG+4.rwksp');

try
    bvr_checkparport('type','S');
catch
    error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];

% prepare settings for classifier training
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.setup= 'LRP';
bbci.clab= {'*'};
bbci.classes= 'auto';
bbci.classDef= {1, 2, 3; 'left','right','foot'};
bbci.feedback= 'first_value';
bbci.setup_opts.ilen_apply= 750;
bbci.withgraphics = 1;
bbci.setup= 'lrp';
bbci.setup_opts.model= {'RLDAshrink', 'scaling',1, 'store_means',1};
bbci.setup_opts.clab = {'*'}; %{'F3,4,5,6','FC5-6','C5-6','CP5-6','P5,1,2,6'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_carlrp_fullChans');
bbci.adaptation.running= 0;

global FVREF; % remembers baselining values during bbci_bet_apply (online)

global bbci_default
bbci_default= bbci;

cfy_name= 'cls_LRP_mcc_VPfbc.mat';% This classifier is used during first stage
copyfile([EEG_RAW_DIR '/pretrained_LRP_classifiers/' cfy_name '*'],TODAY_DIR); % copy classifier to TODAY_DIR
path([BCI_DIR 'acquisition/setups/season11'], path);


%% - Artefaktmessung: : Test
fprintf('\n\nArtifact test run.\n');
%[seq, wav, opt]= setup_season11_artifacts_demo('clstag', 'LRF');
[seq, wav, opt]= setup_season11_artifacts_demo('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT TEST measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt, 'test',1);


%% - Artefaktmessung: Augenbewegungen, Ruhemessung mit Augen auf und Augen zu
fprintf('\n\nArtifact recording.\n');
%[seq, wav, opt]= setup_season11_artifacts('clstag', 'LRF');
[seq, wav, opt]= setup_season11_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt,'filename', 'arte_amAnfang');




%% Init: Start Pyff
PYFF_DIR = 'C:\svn\pyff\src';
general_port_fields= struct('bvmachine','127.0.0.1',...
    'control',{{'127.0.0.1',12471,12487}},...
    'graphic',{{'',12487}});

general_port_fields.feedback_receiver= 'pyff';

PyffStarted = pyff('startup', 'gui',1, 'dir',PYFF_DIR); % Start PYFF
pause(5);

pyff('init','FeedbackCursorArrow2'); % Init Feedback
pause(2);
pyff('play'); % Start the Feedback (it will be in pause mode) 

% PYFF settings
fb_opt_int.trials = 30;
fb_opt_int.durationPerTrial = 5000;
fb_opt_int.hitMissDuration =  5000; % Pause zwischen 2 Trials
fb_opt_int.countdownFrom = 5;


%pyff('set', fb_opt);   % Send Settings, uncomment if you want to set non
                        % integer values
pyff('setint', fb_opt_int); % Send Settings (Interger Values)
fb_opt= []; fb_opt_int= [];

fprintf('Position feedback window on screen\n Remember VP_CODE and setupfile\\n');


%% General Settings
all_classes= {'left', 'right', 'foot'};
bbci_cfy= [TODAY_DIR cfy_name];            % Set Classifier to use to the pretrained one
savename{1} = '/imag_fbarrow_LRPprecalc_'; % Filenamen für feedback daten, die mit vortrainiertem cls aufgenommen wurden
savename{2} = '/imag_fbarrow_LRPopt_';     % Filenamen für feedback daten, die mit cls aufgenommen wurde, der trainiert wurde mit daten, die am gleichen Tag gemessenen wurden
savename{3} = '/exec_fbarrow_LRPopt_';     % Filenamen für ausgeführte Bewegungen

bbci.start_marker = 210;
bbci.quit_marker = 254;


%% Init and record calibration data (see run_season11.m)
bvr_startrecording(['impedances_beginning' VP_CODE]);
pause(1);
bvr_sendcommand('stoprecording');


%% - Run Stage 1

runs = 1;
fb_opt_int.trials = 40; % Number of trails
pyff('setint', fb_opt_int); % Send Settings (Integer Values)

fb_opt.g_rel = 0.25; % Staerke der Kreuzbewegung
%fb_opt.g_rel = 0.0; % Staerke der Kreuzbewegung
%fb_opt.bias = 1.0

CLSTAG = 'LR';

for i=1:runs
    ci1= find(CLSTAG(1)=='LRF'); ci2= find(CLSTAG(2)=='LRF');
    classes= all_classes([ci1 ci2]);

    fprintf('Starting Run with classes %s and %s.\n', char(classes(1)), char(classes(2)));

    fb_opt.pause= false;
    fb_opt.countdown= true;
    fb_opt.availableDirections= {char(classes(1)),char(classes(2))};
    
    pyff('set', fb_opt);

    bvr_startrecording([savename{1}, CLSTAG],  'impedances',0);
    bbci_bet_apply(bbci_cfy,'bbci.feedback','first_value', 'bbci.fb_port', 12345, 'bbci.start_marker',bbci.start_marker, 'bbci.quit_marker',bbci.quit_marker);
    
    fprintf('Run Completed. \nEnter RETURN to continue\n');
    pause

end



%% - Run Stage 1b: Executed Movements

runs = 1;
fb_opt_int.trials = 40; % Number of trails
pyff('setint', fb_opt_int); % Send Settings (Integer Values)

%fb_opt.g_rel = 0.25; % Staerke der Kreuzbewegung
fb_opt.g_rel = 0.0; % Staerke der Kreuzbewegung
%fb_opt.bias = 1.0

CLSTAG = 'LR';

for i=1:runs
    ci1= find(CLSTAG(1)=='LRF'); ci2= find(CLSTAG(2)=='LRF');
    classes= all_classes([ci1 ci2]);

    fprintf('Starting Run with classes %s and %s.\n', char(classes(1)), char(classes(2)));

    fb_opt.pause= false;
    fb_opt.countdown= true;
    fb_opt.availableDirections= {char(classes(1)),char(classes(2))};
    
    pyff('set', fb_opt);

    bvr_startrecording([savename{3}, CLSTAG],  'impedances',0);
    bbci_bet_apply(bbci_cfy,'bbci.feedback','first_value', 'bbci.fb_port', 12345, 'bbci.start_marker',bbci.start_marker, 'bbci.quit_marker',bbci.quit_marker);
    
    fprintf('Run Completed. \nEnter RETURN to continue\n');
    pause

end



%% analyze

bbci.train_file= strcat(bbci.subdir, [char(savename(1)),'*']);

bbci_bet_prepare;
Cnt = proc_selectChannels(Cnt,'not','EMG*');
bbci_bet_analyze;

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard;
bbci_bet_finish;

bbci_cfy_2= bbci.save_name;  % Name des neuen Cls
FVREF = []; % For secound "classifier", takes care of LRP Baselining 

%% - Run Stage 2


runs = 1;
fb_opt_int.trials = 20; % Number of trails
fb_opt.g_rel = 0.5; %
%fb_opt.bias = 1.0

pyff('setint', fb_opt_int); % Send Settings (Integer Values)

CLSTAG = 'LR';
for i=1:runs
    ci1= find(CLSTAG(1)=='LRF'); ci2= find(CLSTAG(2)=='LRF');
    classes= all_classes([ci1 ci2]);

    fprintf('Starting Run with classes %s and %s.\n', char(classes(1)), char(classes(2)));
    fb_opt.pause= false;
    fb_opt.countdown= true;
    fb_opt.availableDirections= {char(classes(1)),char(classes(2))};
    pyff('set', fb_opt);   
    bvr_startrecording([savename{2}, CLSTAG],  'impedances',0);
    bbci_bet_apply(bbci_cfy_2,'bbci.feedback','first_value', 'bbci.fb_port', 12345, 'bbci.start_marker',bbci.start_marker, 'bbci.quit_marker',bbci.quit_marker);
    fprintf('Run Complete\nEnter dbcont to continue\n');
    keyboard;
end

%% - Artefaktmessung: Augenbewegungen, Ruhemessung mit Augen auf und Augen zu
fprintf('\n\nArtifact recording.\n');
%[seq, wav, opt]= setup_season11_artifacts('clstag', 'LRF');
[seq, wav, opt]= setup_season11_artifacts('clstag', '');
fprintf('Press <RETURN> when ready to start ARTIFACT measurement.\n');
pause
stim_artifactMeasurement(seq, wav, opt,'filename', 'arte_amEnde');


%% ESTIM
disp('ESTIM vorbereiten.')
disp('Verbinde ParallelportStimulus und E-Stim ueber RESPONSE Stecker!')
disp('(Wenn fertig, RETURN drücken fuer einen Test...')
pause
for i=1:80
    ppTrigger(32);
    pause(2);
end

disp('ESTIM Medianus LINKS vorbereiten.')
disp('Verbinde ParallelportStimulus und E-Stim ueber RESPONSE Stecker!')
disp('(Wenn fertig, RETURN drücken für Aufnahme)')
pause
filename= bvr_startrecording('Stimulation_NervusMedianus_linkeHand',  'impedances',0);
pause(5)
for i=1:100
    ppTrigger(32);
    pause(2);
end
bvr_sendcommand('stoprecording');

for i=1:40
    ppTrigger(32);
    pause(2);
end

disp('ESTIM Medianus RECHTS vorbereiten.')
disp('Verbinde ParallelportStimulus und E-Stim ueber RESPONSE Stecker!')
disp('(Wenn fertig, RETURN drücken fuer Aufnahme.)')
pause
filename= bvr_startrecording('Stimulation_NervusMedianus_rechteHand',  'impedances',0);
pause(5)
for i=1:100
    ppTrigger(32);
    pause(2);
end
bvr_sendcommand('stoprecording');

