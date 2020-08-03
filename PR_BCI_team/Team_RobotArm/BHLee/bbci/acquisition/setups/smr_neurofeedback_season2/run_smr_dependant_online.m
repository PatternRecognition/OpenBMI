%% Preparation of the EEG cap
bvr_sendcommand('checkimpedances');
stimutil_waitForInput('msg_next','when finished preparing the cap.');
bvr_sendcommand('viewsignals');
pause(5);

%% Relaxation - eyes open changed to 5 min durchgehend
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax_long;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Train SMR Extractor on data of relaxation
bbci_nf= [];
bbci_nf.setup= 'smr_extractor';
bbci_nf.func_mrk= 'durchrauschen';
bbci_nf.train_file= strcat(TODAY_DIR, 'resting', VP_CODE);
bbci_nf.save_name= strcat(TODAY_DIR, 'bbci_smr_extractor');
bbci_nf.feedback= '';
bbci_nf.setup_opts.extractor_fcn= 'smr_extractor_adaptCentralBand';
bbci_nf.setup_opts.band_search = 'alpha+beta';
bbci_nf.setup_opts.spat_flt='lar';
bbci_nf.quit_marker= 254;

bbci= bbci_nf; 
bbci_bet_prepare
mrk_orig= mrk;
bbci_bet_analyze

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish
%bbci_nf= bbci;

%% NEUROFEEDBACK starting up
desc= stimutil_readDescription('SMR_Neurofeedback');
h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
%% ** Startup pyff **
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks\SMR_NeuroFeedback" &']);
bvr_sendcommand('viewsignals');
pause(8)
general_port_fields.bvmachine= '127.0.0.1';
general_port_fields.control{3}= 12345;
general_port_fields.feedback_receiver= 'pyff';
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

%% Neurofeedback of SMR amplitude - Run 1
%send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
send_xmlcmd_udp('interaction-signal', 'i:trialTime',5*60);
stimutil_waitForInput('msg_next','to start Run 1 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
%bbci_bet_apply(bbci.save_name, 'modifications',{'bbci.fb_port', 12345});
bbci_bet_apply(bbci_nf.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

%% Impedance measurement
pause(1);
fprintf('\n\nStarting impedance measurement.\n');
pause(3);
bvr_sendcommand('startimprecording', ['impedances_before_feedback.eeg']);
pause(15);
bvr_sendcommand('stoprecording');


%% Prepare CSP based feedback
setup_file= 'season10\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= ['imag_fbarrow_LapC3z4'];
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [EEG_RAW_DIR 'subject_independent_classifiers/season10/Lap_C3z4_bp2_LR'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'adaptation.policy', 'pcovmean', ...
                'adaptation.running', 1, ...
                'adaptation.adaptation_ival', [1500 4000], ...
                'adaptation.load_tmp_classifier', 0};
settings_fb= struct('trials_per_run', 80, ...
                    'duration_show_selected', 3000, ...
                    'remove_cue_at_end_of_trial', 1);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);

%% CALIBRATION
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15] at Lap C3,4), pcovmean adaptation

fprintf('Press <RETURN> when ready to start the next task.\n');
pause
%set_general_port_fields({'tubbci5','tubbci5'})
general_port_fields.feedback_receiver= 'matlab';
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(15)
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start Run 1 and wait (press <RETURN only once!).\n');
desc= stimutil_readDescription('SMR_dependant_online_imag_arrow_cb');
h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup, 'impedances',0);
fprintf('Press <RETURN> when feedback has finished.\n');
pause


%% - Train CSP-based classifier on Feedback Run 1
bbci_csp= bbci_default;
bbci_csp.setup= 'cspauto';
bbci_csp.train_file= strcat(bbci_csp.subdir, '/imag_fbarrow_LapC3z4',VP_CODE);
bbci_csp.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_24chans');
bbci_csp.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1, 'store_extinvcov',1};
bbci_csp.setup_opts.clab= {'F3,4','FC5,1,2,6','C5-6','CCP5,3,4,6','CP3,z,4','P5,1,2,6'};
bbci_csp.adaptation.load_tmp_classifier= 0;

bbci= bbci_csp;
bbci_bet_prepare
bbci_bet_analyze

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish
bbci_csp_output= bbci;

%-newblock
%% - BBCI adaptive Feedback, CSP-based classifier, pcovmean adaptation
fprintf('Press <RETURN> when ready to start the next task.\n');
pause
setup.general.savestring= 'imag_fbarrow_pcovmean';
setup.control_player1.bbci.adaptation.adaptation_ival= min(bbci_csp_output.analyze.ival, [2000 4000]);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_24chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(15)
nogui_send_setup(setup);
fprintf('Press <RETURN> when ready to start Run 2 and wait (press <RETURN only once!).\n');
desc= stimutil_readDescription('SMR_dependant_online_imag_arrow_cb');
h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup, 'impedances',0);
fprintf('Press <RETURN> when feedback has finished.\n');
pause

%-newblock
%% - Train CSP-based classifier on Feedback Runs 1, 2
bbci_csp.train_file= strcat(bbci_csp.subdir, '/imag_fbarrow_*');
bbci_csp.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci_csp.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci_csp.adaptation.load_tmp_classifier= 1;

bbci = bbci_csp;
bbci_bet_prepare
bbci_bet_analyze

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish
bbci_csp_output = bbci;

%% Relaxation - eyes open run 2 (5min durchgehend)
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax_long;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

%% Train SMR Extractor on data of resting with bands from calibration
bbci_nf.train_file= strcat(TODAY_DIR, 'resting', VP_CODE, '02');
bbci_nf.save_name= strcat(TODAY_DIR, 'bbci_smr_extractor_run2');
bbci_nf.setup_opts.band= analyze.band; 

bbci= bbci_nf;
bbci_bet_prepare
mrk_orig= mrk;
bbci_bet_analyze

fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci_bet_finish
%bbci_nf= bbci;

%% Neurofeedback RUN 2
desc= stimutil_readDescription('SMR_Neurofeedback');
h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks\SMR_NeuroFeedback" &']);
bvr_sendcommand('viewsignals');
pause(8)
general_port_fields.feedback_receiver= 'pyff';
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
send_xmlcmd_udp('interaction-signal', 'i:trialTime',5*60);
stimutil_waitForInput('msg_next','to start Run 2 of SMR NeuroFeedback Training.');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci_nf.save_name);
pause(3);
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%% Combine classifiers (run1 of the feedback)
bbci_csp.save_name_csp_only= bbci_csp.save_name;
bbci_csp.save_name= [bbci_csp.save_name_csp_only '_plus_nf'];
S= load(bbci_csp.save_name_csp_only);
S_NF= load(bbci_nf.save_name);
S.cont_proc(2)= S_NF.cont_proc(1);
S.feature(2)= S_NF.feature(1);
S.feature(2).cnt= 2;
S.cls(2)= S_NF.cls(2); 
save(bbci_csp.save_name, '-STRUCT','S');

setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
bbci_cfy= [bbci_csp.save_name];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'adaptation.policy', 'pmean', ...
                'adaptation.adaptation_ival', min(bbci_csp_output.analyze.ival, [2000 4000]), ...
                'adaptation.load_tmp_classifier', 1, ...
                'feedback',''};
settings_fb= struct('type', 'feedback_cursor_arrow_smr_nf', ...
                    'duration_show_selected', 3000, ...
                    'remove_cue_at_end_of_trial', 1, ...
        'trials_per_run', 30, ...
        'duration_smr_wait', 20000, ...
        'duration_smr_descend', 10000, ...
        'break_every', 15, ...
        'smr_range', [NaN NaN], ...
        'smr_undershoot', 0, ...
        'smr_thresh', 1);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);

undershoot_list= [0 1; 0 0; 1 0];
filename_range= 'd:\svn\bbci\acquisition\setups\smr_neurofeedback_season2\smr_range';

for run_pair= 1:3,
  %% loading the smr_range file for starting feednback run 1
  movefile([filename_range '.txt'], [filename_range int2str(run_pair) '.txt']);
  smr_val=load([filename_range int2str(run_pair) '.txt']);
  
  %% Feedback Run 1  - BBCI adaptive Feedback, CSP-based classifier, pmean
  %% adaptation RUN 1
  fprintf('Press <RETURN> when ready to start the next task.\n');
  pause
  general_port_fields.feedback_receiver= 'matlab';
  settings_fb.smr_range= round(smr_val*100)/100;
  settings_fb.smr_undershoot= undershoot_list(run_pair,1);
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(15)
  nogui_send_setup(setup);
  fprintf('Press <RETURN> when ready to start CSP+SMR-NF Run %d and wait (press <RETURN only once!).\n', 2*run_pair-1);
  desc= stimutil_readDescription('SMR_dependant_online_imag_arrow_fb');
  h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
  nogui_start_feedback(setup, 'impedances',0);
  fprintf('Press <RETURN> when feedback has finished.\n');
  pause
  
  %% Feedback Run 2  - BBCI adaptive Feedback, CSP-based classifier, pmean
  %% adaptation RUN 2
  fprintf('Press <RETURN> when ready to start the next task.\n');
  pause
  settings_fb.smr_undershoot= undershoot_list(run_pair,2);
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
  pause(7)
  nogui_send_setup(setup);
  fprintf('Press <RETURN> when ready to start CSP+SMR-NF Run %d and wait (press <RETURN only once!).\n', 2*run_pair);
  desc= stimutil_readDescription('SMR_dependant_online_imag_arrow_fb');
  h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
  nogui_start_feedback(setup, 'impedances',0);
  fprintf('Press <RETURN> when feedback has finished.\n');
  pause

  if run_pair<3,
    %% Neurofeedback Run 3
    desc= stimutil_readDescription('SMR_Neurofeedback');
    h_desc= stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
    system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks\SMR_NeuroFeedback" &']);
    bvr_sendcommand('viewsignals');
    pause(8)
    general_port_fields.feedback_receiver= 'pyff';
    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit');
    send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','smr_neurofeedback');
    send_xmlcmd_udp('interaction-signal', 'i:trialTime',3*60);
    send_xmlcmd_udp('interaction-signal', 'i:screenPos',[1924 20], 'i:screenSize',[1910, 1175]);
    stimutil_waitForInput('msg_next','to start Run 3 of SMR NeuroFeedback Training.');
    send_xmlcmd_udp('interaction-signal', 'command', 'play');
    bbci_bet_apply(bbci_nf.save_name);
    pause(3);
    send_xmlcmd_udp('interaction-signal', 'command', 'quit');
  end
end

%% Impedance measurement
pause(1);
fprintf('\n\nStarting impedance measurement.\n');
pause(3);
bvr_sendcommand('startimprecording', ['impedances_after_feedback.eeg']);
pause(15);
bvr_sendcommand('stoprecording');

%% Relaxation - eyes open
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_smr_neurofeedback_relax_long;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);

fprintf('End of this session\n');
