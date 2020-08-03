global RAW_ETH;

% fprintf('Welcome to session 1 of calibration support experiment!\n');
% 
% bvr_sendcommand('checkimpedances');
% fprintf('Prepare cap. Press <RETURN> when finished.\n');
% pause


%% TEST FES
RAW_ETH = 1; 
[stim, opt] = setup_mundus_calibsup_image_arrow({'left', 'right','foot'}, 5, 2000, 3000);

fprintf('TEST: Press <RETURN> when ready to start ''nmes'' test 1.\n');
pause
stim_visualCuesExt(stim, opt, 'test',1, 'test_mode', 1, 'fes', 1);
% % 
% % %% TEST MI

RAW_ETH = 0; 
[stim, opt] = setup_mundus_calibsup_image_arrow({'left','right','down'}, 10, 2000, 3000);

fprintf('TEST: Press <RETURN> when ready to start ''imagined movements'' test.\n');
pause
stim_visualCuesExt(stim, opt, 'test',1, 'test_mode', 1);

% % 
% %% PART 1
% %--------------------------------------------------------------------------
% 
% %% MI CALIB RUN

RAW_ETH = 0;
[stim, opt] = setup_mundus_calibsup_image_arrow({'right','down'}, 40, 2000, 3000);
opt.filename = 'FES_cb';

for i=1:2
   fprintf('Press <RETURN> when ready to start ''MI'' measurement.\n');
   pause
   stim_visualCuesExt(stim, opt);
end;
fprintf('end of only MI runs.\n');

% %% data analysis to select the best 2 classes.
%bbci.classDef= { 1, 3;  'left', 'foot'};
bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci.setup='cspauto';
bbci.subdir = 'D:\data\bbciRaw\VPkn_08_08_22';
bbci.train_file= strcat(bbci.subdir, '/imag_arrow*');

%bbci.impedance_threshold = Inf; % for testing purposes!!!!!

bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed.\n');
keyboard
close all

bbci.adaptation.running= 1;
bbci.adaptation.policy= 'pmean';
bbci.adaptation.adaptation_ival= bbci.setup_opts.ival;
bbci.adaptation.UC= 0.05;
bbci_bet_finish


% FEEDBACK



setup_file= 'season10\cursor_adapt_pmean.setup';
setup= nogui_load_setup(setup_file);

setup.general.savestring= 'imag_fbarrow_pmean_fes';
cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
bbci_cfy= [TODAY_DIR 'bbci_classifier_cspauto_48chans']; %check!!
cmd_bbci= ['dbstop if error; global BBCI_VERBOSE; BBCI_VERBOSE=1; dbstop in bbci_bet_apply at 317; global OFFLINE_TEST; OFFLINE_TEST = 1;bbci_bet_apply(''' bbci_cfy ''');'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];
%cmd_bbci= ['dbstop if error; bbci_bet_apply();'];
S= load(bbci_cfy);

classes= S.bbci.classes;youtube
CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
ri = 1;

for ri= 1:3,  
%    system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
%   pause(15)
%   settings_bbci= {'start_marker', 210, ...
%                   'quit_marker', 254, ...
%                   'feedback', '1d', ...
%                   'fb_port', 12345, ...
%                   'adaptation.mrk_start', [ci1 ci2], ...
%                   'adaptation.load_tmp_classifier', ri>1};
%      settings_fb= {'classes', classes, ...
%                 'trigger_classes_list', {'right','foot'},  ...
%                 'duration_before_free', S.bbci.analyze.ival(1), ...
%                 'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
%             
%      setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
%      setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
%      setup.graphic_player1.machine = '127.0.0.1';
%                  
%      nogui_send_setup(setup);
%      
%      fprintf('Press <RETURN> when ready to start Run %d and wait (press <RETURN only once!).\n', 4+ri);
%      desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
%      stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
%      nogui_start_feedback(setup, 'impedance', 0);
%      fprintf('\nPress <RETURN>.\n');
%      pause;
%      fprintf('Thank you for letting me know ...\n');
     
     system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']); 
  pause(15)
  settings_bbci= {'start_marker', 210, ...
                  'quit_marker', 254, ...
                  'feedback', '1d', ...
                  'fb_port', 12345, ...
                  'adaptation.mrk_start', [ci1 ci2], ...
                  'adaptation.load_tmp_classifier', ri>1};
     settings_fb= {'classes', classes, ...
                'trigger_classes_list', {'left','right','foot'},  ...
                'duration_before_free', S.bbci.analyze.ival(1), ...
                'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
     setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
     setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
     nogui_send_setup(setup);
     fprintf('Press <RETURN> when ready to start Run %d and wait (press <RETURN only once!).\n', 4+ri);
     desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
     stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
     nogui_start_feedback(setup, 'impedance', 0);
     fprintf('\nPress <RETURN>.\n');
     pause;
     fprintf('Thank you for letting me know ...\n');

end

%% MI + FES RUN

RAW_ETH = 1;
[stim, opt] = setup_mundus_calibsup_image_arrow({'right','down'}, 25, 2000, 3000);
opt.filename = 'FES_cb';

for i=1:3
   fprintf('Press <RETURN> when ready to start ''electrical stimulation'' measurement.\n');
   pause
   stim_visualCuesExt(stim, opt);
end;
fprintf('end of FES runs.\n');


%% PART 2
%--------------------------------------------------------------------------
RAW_ETH = 1;
[stim, opt] = setup_mundus_calibsup_image_arrow(bbci.classes, 40, 2000, 3000);
opt.filename = 'MI_FES_worstlimb';

for i=1:2
    fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
    pause
    stim_visualCuesExt(stim, opt);
end;

fprintf('end of MI + FES runs.\n'); 

opt.filename = 'MI_FES_samelimb';
for i=1:2
   fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
   pause
   stim_visualCuesExt(stim, opt);
end;