
if ~exist('TODAY_DIR','var')
  startup_bbcilaptop05
end

if isempty(TODAY_DIR)
  setup_labrotation09_simon
end

monitor_pos = get(0,'MonitorPosition');
prim_mon = monitor_pos(1,:);
VP_SCREEN = prim_mon;

testing = 0;
cls_ival = [-300 -200];
keypress_tolerance = 50;
fprintf('Using a classification window of [%i,%i] relative to keypress.\n', cls_ival)
fprintf('Using a keypress tolerance of +/- %ims\n', keypress_tolerance)
if testing 
  disp('Test run (i.e. data will be saved, but less trials per block and no impedance checking)!') 
end

if ~testing
  artefact_measurement = 0;
  nTestBlocks = 0;
  nTrainBlocks_none = 0;
  nTrainBlocks_random = 3;
  nFeedbackBlocks = 6;
  
  bvr_sendcommand('checkimpedances');
  fprintf('Prepare cap. Press <RETURN> when finished.\n');
  pause
else
  artefact_measurement = 0;
  nTestBlocks = 0;
  nTrainBlocks_none = 1;
  nTrainBlocks_random = 1;
  nFeedbackBlocks = 1;
end

system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins --additional-feedback-path=D:\svn\pyff_external_feedbacks" &');
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);


if artefact_measurement
  %% - Artifact measurementj
  art_opt.language = 'german';
  fprintf('\n\nArtifact test run.\n');
  fprintf('Default language: %s\n', art_opt.language)
  keyboard
  [seq, wav, opt]= setup_labrotation09_simon_artifacts_demo(art_opt);
  fprintf('Press <RETURN> when ready to start artifact measurement test.\n');
  pause
  stim_artifactMeasurement(seq, wav, opt, 'test',1);

  %-newblock
  fprintf('\n\nArtifact recording.\n');
  [seq, wav, opt]= setup_labrotation09_simon_artifacts(art_opt);
  fprintf('Press <RETURN> when ready to start artifact measurement.\n');
  pause
  stim_artifactMeasurement(seq, wav, opt);
  fprintf('Switch off the speakers.\nPress <RETURN> when ready to go to the feedback runs.\n');
  pause
  close all
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for block = 1:nTestBlocks
  curr_block = 'test';
  if block <= ceil(nTestBlocks/2)
    showClassifier = 'none';
  else
    showClassifier = 'random';
  end
  fprintf(['Press <return> to start the next test block (showClassifier="' showClassifier '")\n']);
  pause
  if testing
    nTrials = 10;
  else
    nTrials = 7;
  end
eval('exp_block')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING BLOCKS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for block = 1:nTrainBlocks_none+nTrainBlocks_random
  curr_block = 'train';
  fprintf('Press <return> to start the next training block.\n');
  pause
  if block<=nTrainBlocks_none
    showClassifier = 'none';
  else
    showClassifier = 'random';
  end
  if testing
    nTrials = 5;
  else
    nTrials = 50;
  end
  exp_block
end


%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN CLASSFIER %%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nTrain classifier...\n');
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
setup = 'erp';   %  'cspauto' / 'erp'
bbci= []; bbci.setup_opts = [];
bbci.setup_opts_erp = []; bbci.setup_opts_csp = [];
bbci = set_defaults(bbci, 'setup', setup, ...
                          'train_file', strcat(subdir, '\lc_*'), ... 
                          'clab', {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'}, ...
                          'classDef', {{'noKP'}, {'KP'}; 'noKP', 'KP'}, ...
                          'func_mrk', 'mrk_addnokeypress', ...
                          'feedback', '1d', ...
                          'save_name', strcat(TODAY_DIR, 'lc_classifier_', setup), ...
                          'setup_file', strcat(TODAY_DIR, 'lc_classifier_', setup, '_setup_001'), ...
                          'start_marker', 252, ...
                          'quit_marker', 253, ...
                          'withgraphics', 0, ...
                          'withclassification', 1);
                        
bbci.setup_opts_erp = set_defaults(bbci.setup_opts_erp, 'model', {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1}, ...
                                                'clab', {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'}, ...
                                                ...%'check_ival', [-2000 500],...
                                                'ival', [-2500 500], ...  
                                                'xTrials', [2 5], ...   
                                                'erp_baseline', 500, ... 
                                                'erp_winsize', 1500, ...
                                                'winends_tr', 0, ...
                                                'winends_te', -100:50:100, ...
                                                'erp_band', [1.5 3.5], ...                                                
                                                'visu_band', [1.5 3.5], ...
                                                'visu_ival', [-1500 500], ...
                                                'func_adaptBias', [], ...% 'adapt_bias', ...    % script to be executed for adapting the bias of the classifier
                                                'laplace', 0);     
   
bbci.setup_opts_csp = set_defaults(bbci.setup_opts_csp, 'model', {'RLDAshrink', 'gamma',0, 'scaling',1, 'store_means',1}, ...
                                                'clab', {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'}, ...
                                                'check_ival', [-2000 500],...
                                                'default_ival', [-400 0],...
                                                'visu_ival', [-1500 500], ...
                                                'ival', [-400 0], ...%[-300 150], ...
                                                'visu_band', [10 25], ...
                                                'min_ival_length', 50, ...
                                                'enlarge_ival_append', 'start', ...
                                                'func_adaptBias', [], ...%'adapt_bias', ...    % script to be executed for adapting the bias of the classifier
                                                'verbose', 2, ...
                                                'laplace', 0, ...
                                                'usedPat', 'auto');     %If'auto' mode does not work robustly:
                                                                        %bbci.setup_opts.usedPat= [1:6]; 
                                                                        
if strcmpi(setup,'erp')
    bbci.setup_opts = bbci.setup_opts_erp;
elseif strcmpi(setup,'csp')
    bbci.setup_opts = bbci.setup_opts_csp;
else
    error('unknown setup.')
end

if testing
  bbci.impedance_threshold = [];
  bbci.reject_artifacts = 0;
  bbci.reject_channels = 0;
end 
                                          
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to save classifier and proceed. To change the setup to be used, edit bbci.setup_file.\n');
keyboard
bbci_bet_finish
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% FEEDBACK BLOCKS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Start Feedback\n');
for block = 1:nFeedbackBlocks
  curr_block = 'feedback';
  fprintf(['Check if marker file .vhmrk was written correctly. \n', ...
           'Press <return> to start the next feedback block.\n']);
  pause
  if testing
    nTrials = 5;
  else
    nTrials = 50;
  end
  exp_block_feedback
end

