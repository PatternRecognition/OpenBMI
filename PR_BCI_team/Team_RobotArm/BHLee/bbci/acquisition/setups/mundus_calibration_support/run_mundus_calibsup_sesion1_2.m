% global RAW_ETH;
% %ETH_MARKERS = 1;
% 
% fprintf('Welcome to session 1 of calibration support experiment!\n');
% 
% % bvr_sendcommand('checkimpedances');
% % fprintf('Prepare cap. Press <RETURN> when finished.\n');
% % pause
%  
%  
n_runs=2;
RAW_ETH = 0;
% bbci.setup='adaptationstudy_season2';
% 
% % TEST

% [stim, opt] = setup_mundus_calibsup_image_arrow({'left','right','down'});
% fprintf('TEST: Press <RETURN> when ready to start ''imagined movements'' test.\n');
% pause
%  
% stim_visualCuesExt(stim, opt, 'test',1);
% 
% 
% % PART 1
% %--------------------------------------------------------------------------
% % MI RUN
% opt.filename = 'MI_ONLY';
% 
% for i=1:n_runs
%     %fprintf('Press <RETURN> when ready to start the recording.\n');
%     %pause
%     %setup_vitalbci_season1_imag_arrow;
%     fprintf('Press <RETURN> when ready to start ''imagined movement'' measurement.\n');
%     pause
%     stim_visualCuesExt(stim, opt);
% end;
% fprintf('end of MIruns.\n'); 
% 
% % FES RUN
% RAW_ETH = 1;
% opt.filename = 'FES_ONLY';
% for i=1:n_runs
%    %setup_vitalbci_season1_imag_arrow;
%    fprintf('Press <RETURN> when ready to start ''electrical stimulation'' measurement.\n');
%    pause
%    stim_visualCuesExt(stim, opt);
% end;
% fprintf('end of only FES runs.\n');
% % MI + FES RUN
% 
% opt.filename = 'MI_FES';
% for i=1:n_runs
%    %setup_vitalbci_season1_imag_arrow;
%    fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
%    pause
%    stim_visualCuesExt(stim, opt);
% end;
% fprintf('end of MI + FES runs.\n'); 
% %data analysis to select the best 2 classes.

bbci.setup_opts.model= {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1};
bbci.setup_opts.clab= {'F3-4','FC5-6','CFC5-6','C5-6','CCP5-6','CP5-6','P5-6','PO3,z,4'};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_cspauto_48chans');
bbci.setup='cspauto';
bbci.train_file= strcat(bbci.subdir, '/MI_ON*');

%--------------------------------------------------------------------------
bbci.impedance_threshold = Inf; % for testing purposes!!!!!
%--------------------------------------------------------------------------

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



% PART 2
%--------------------------------------------------------------------------
RAW_ETH = 1;
[stim, opt] = setup_mundus_calibsup_image_arrow(bbci.classes);
opt.filename = 'MI_FES_fb';
for i=1:n_runs-1
    %setup_vitalbci_season1_imag_arrow;
    fprintf('Stimulation of the limb not used for MI');
    fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
    pause
    stim_visualCuesExt(stim, opt);
 end;
% fprintf('end of MI + FES runs.\n'); 

% opt.filename = 'MI_FES';
% for i=1:n_runs-1
%    %setup_vitalbci_season1_imag_arrow;
%    fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
%    pause
%    stim_visualCuesExt(stim, opt);
% end;
% fprintf('end of MI + FES runs.\n'); 

% opt.filename = 'MI_FES';
% for i=1:n_runs-1
%    %setup_vitalbci_season1_imag_arrow;
%    fprintf('Press <RETURN> when ready to start ''imagined movement + electrical stimulation'' measurement.\n');
%    pause
%    stim_visualCuesExt(stim, opt);
% end;
% fprintf('end of MI + FES runs.\n'); 


% FEEDBACK
% setup_file= 'season10\cursor_adapt_pmean.setup';
% setup= nogui_load_setup(setup_file);
% setup.general.savestring= 'imag_fbarrow_pmean';
% cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s''; VP_SCREEN= [%d %d %d %d];', VP_CODE, TODAY_DIR, VP_SCREEN);
% bbci_cfy= [TODAY_DIR '/bbci_classifier_cspauto_48chans']; %check!!
% cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''');'];

% eth init
%cmd_bbci = ['fprintf(''Press <RETURN> when ready to start ''imagined movements'' test.\n''); pause; rawethmex(''init'', ''name'', ''00:E0:81:78:34:AC'', 1, 1, 1);' cmd_bbci]
% cmd_bbci = ['rawethmex(''init'', ''{AC68C2CB-C272-4C12-A7D8-732C6B54967B}'', ''5C:FF:35:09:49:EC'', 1, 1, 1);' cmd_bbci]
% S= load(bbci_cfy);
% classes= S.bbci.classes;
% CLSTAG= [upper(classes{1}(1)) upper(classes{2}(1))];
% ci1= find(CLSTAG(1)=='LRF');
% ci2= find(CLSTAG(2)=='LRF');
% 
% for ri= 1:n_fb_runs,  
%   system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);
%   pause(15)
%   settings_bbci= {'start_marker', 210, ...
%                   'quit_marker', 254, ...
%                   'feedback', '1d', ...
%                   'fb_port', 12345, ...
%                   'adaptation.mrk_start', [ci1 ci2], ...
%                   'adaptation.load_tmp_classifier', ri>1};
%      settings_fb= {'classes', classes, ...
%                 'trigger_classes_list', {'left','right','foot'},  ...
%                 'duration_before_free', S.bbci.analyze.ival(1), ...
%                 'pause_msg', [CLSTAG(1) ' vs. ' CLSTAG(2)]};
%      setup.graphic_player1.feedback_opt= overwrite_fields(setup.graphic_player1.feedback_opt, settings_fb);
%      setup.control_player1.bbci= overwrite_fields(setup.control_player1.bbci, settings_bbci);
%      nogui_send_setup(setup);
%      fprintf('Press <RETURN> when ready to start Run %d and wait (press <RETURN only once!).\n', 4+ri);
%      desc= stimutil_readDescription(['season10_imag_fbarrow_' CLSTAG]);
%      stimutil_showDescription(desc, 'clf',1, 'waitfor','key', 'delete','fig');
%      nogui_start_feedback(setup);
%      fprintf('\nPress <RETURN>.\n');
%      pause;
%      fprintf('Thank you for letting me know ...\n');
% end