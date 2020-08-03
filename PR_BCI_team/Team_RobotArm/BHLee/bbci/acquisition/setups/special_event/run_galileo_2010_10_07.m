%% - Run 1
%% - BBCI adaptive Feedback (subject-independent classifier, log-bp[8-15;16-35] at Lap C3,4), pcovmean adaptation
desc= stimutil_readDescription('season10_imag_fbarrow_LRF');
%stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig', 'waitfor_msg', 'Press <RETURN> to start feedback: ');

nRuns= 3;
setup_file= 'season10\cursor_adapt_pcovmean.setup';
setup= nogui_load_setup(setup_file);
today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
tmpfile= [TODAY_DIR 'adaptation_pcovmean_Lap_' today_str];
tag_list= {'LR', 'LF', 'FR'};
all_classes= {'left', 'right', 'foot'};
for ri= 1:nRuns,
 for ti= 1:length(tag_list),  
  CLSTAG= tag_list{ti};
  cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
  bbci_cfy= [TODAY_DIR '/Lap_C3z4_bp2_' CLSTAG];
  cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
  system(['matlab -nosplash -r "' cmd_init 'setup_galileo_2010_10_07; ' cmd_bbci '; exit &']);
  pause(7)
  ci1= find(CLSTAG(1)=='LRF');
  ci2= find(CLSTAG(2)=='LRF');
  classes= all_classes([ci1 ci2]);
  settings_bbci= {'start_marker', 210, ...
                  'quit_marker', 254, ...
                  'feedback', '1d', ...
                  'fb_port', 12345, ...
                  'adaptation.policy', 'pcovmean', ...
                  'adaptation.offset', 750, ...
                  'adaptation.tmpfile', [tmpfile '_' CLSTAG], ...
                  'adaptation.mrk_start', {ci1, ci2}, ...
                  'adaptation_ival', [750 4000], ...
                  'adaptation.load_tmp_classifier', ri>1};
  settings_fb= {'classes', classes, ...
                'trigger_classes_list', {'left','right','foot'},  ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)], ...
                'trials_per_run', 20, ...
                'break_every', 20};
  setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
  setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
  setup.general.savestring= ['imag_fbarrow_LapC3z4_' CLSTAG];
  nogui_send_setup(setup);
%  fprintf('Press <RETURN> when ready to start Run %d°%d: classes %s and wait (press <RETURN only once!).\n', ri, ti, CLSTAG);
%  pause; fprintf('Ok, starting ...\n');
  fprintf('Starting Run %d°%d: classes %s.\n', ceil(ri/2), ti+3*mod(ri-1,2), CLSTAG);
  if ri+ti>2,
    pause(5);  %% Give time to display class combination, e.g. 'L vs R'
  end
  nogui_start_feedback(setup, 'impedances',0);
  fprintf('Press <RETURN> when feedback has finished (windows must have closed).\n');
  pause; fprintf('Thank you for letting me know ...\n');
 end
end


%% Classifier Training for Run 2: CSP on 24 channels
fprintf('Press <RETURN> when ready to start classifier training.\n');
pause;
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow_Lap*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3,4','FC5,1,2,6','C5-6','CP5-6','P5,1,2,6'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_24chans');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
%bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci.adaptation.adaptation_ival= bbci.setup_opts.ival;
bbci.adaptation.UC= 0.05;
bbci_bet_finish


%% - Run 2
%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_24chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.mrk_start', [ci1 ci2], ...
                'adaptation.load_tmp_classifier', 0};
%                'adaptation_ival', S.bbci.setup_opts.ival};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', S.bbci.analyze.ival(1), ...
              'trials_per_run', 50, ...
              'break_every', 50};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
nogui_send_setup(setup);
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup,'impedances',0);


%% Classifier Training for Run 3: CSP on 48 channels
fprintf('Press <RETURN> when ready to start classifier training.\n');
pause;
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.classes= S.bbci.classes;
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','C5-6','CP5-6','P5-6','PO3-4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
%bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci.adaptation.adaptation_ival= bbci.setup_opts.ival;
bbci.adaptation.UC= 0.05;
bbci_bet_finish


%% - Run 3
%% - BBCI adaptive Feedback, CSP-based classifier, pmean adaptation
setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);
setup.general.savestring= 'imag_fbarrow_pmean';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_48chans'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
S= load(bbci_cfy);
classes= S.bbci.classes;
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
pause(7)
settings_bbci= {'start_marker', 210, ...
                'quit_marker', 254, ...
                'feedback', '1d', ...
                'fb_port', 12345, ...
                'adaptation.mrk_start', [ci1 ci2], ...
                'adaptation.load_tmp_classifier', 0};
%                'adaptation_ival', S.bbci.setup_opts.ival};
settings_fb= {'classes', classes, ...
              'trigger_classes_list', {'left','right','foot'},  ...
              'duration_before_free', S.bbci.analyze.ival(1), ...
              'trials_per_run', 30, ...
              'break_every', 50};
setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
nogui_send_setup(setup);
desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
nogui_start_feedback(setup,'impedances',0);

fprintf('Press <RETURN> when ready to start classifier training.\n');
pause;
bbci= bbci_default;
bbci.setup= 'cspauto';
bbci.classes= S.bbci.classes;
bbci.train_file= strcat(bbci.subdir, '/imag_fbarrow*');
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3-4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_final');
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci.adaptation.adaptation_ival= bbci.setup_opts.ival;
%bbci.adaptation.offset= bbci.setup_opts.ival(1);
bbci.adaptation.UC= 0.05;
bbci_bet_finish

bbci_setup= bbci.save_name;
close all;



%% BBCI-controlled Brain-Pong
general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';
pyff('startup', 'gui',1, 'dir',PYFF_DIR)
fprintf('Select BrainPong2 in Pyff GUI, press Init button and then press <ENTER> here.\n');
pause;
fb_opt= []; fb_opt_int= [];
fb_opt_int.screenPos= VP_SCREEN;
primary_screen= get(0, 'ScreenSize');
fb_opt_int.screenPos(2)= primary_screen(4)-VP_SCREEN(4);
fb_opt.trials= 20;
fb_opt.pauseAfter= 25;
fb_opt.bowlSpeed= 1.1;
fb_opt.g_rel=0.95;
fb_opt.font_path=[PYFF_DIR 'Feedbacks\BrainPong2\pirulen.ttf'];
% fb_opt.jitter= 0.1;  %% -> nicht nur ganz-links, ganz-rechts Positionen
fb_opt_int.countdownFrom= 10;
%pyff('init','BrainPong2');  %% -> us this, if Pyff is started without GUI
pause(2)
pyff('set', fb_opt);
pyff('setint', fb_opt_int);
pyff('play')
% comment the following line, to skip the GUI
system(['matlab -r "dbstop if error; matlab_control_gui(''season10/cursor_adapt_pmean'', ''classifier'',''' bbci_setup ''');" &']);
bbci_bet_apply(bbci_setup, 'bbci.feedback','1d', 'bbci.fb_port', 12345, 'bbci.start_marker',31, 'bbci.quit_marker',101);
pause(3)
pyff('stop');
pyff('quit');

pause


%% BBCI-controlled Pinball
fprintf('Plug in flipper trigger cable.\n');
% Start Matlab GUI in Matlab 2 & Start Online Classifier in Matlab 1
% Do NOT execute those lines separately!!!!! Otherwise the GUI will not
% find the classifier
system(['matlab -r "dbstop if error; matlab_control_gui(''flipper_hardware'', ''classifier'',''' bbci_setup ''');" &']);
bbci_bet_apply(bbci_setup, 'bbci.feedback', '1d', 'bbci.fb_port', 12345);
