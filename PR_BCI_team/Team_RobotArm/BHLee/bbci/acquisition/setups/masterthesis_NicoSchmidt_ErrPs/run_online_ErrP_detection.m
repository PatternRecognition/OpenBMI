%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n'), pause

%% Basic settings
nVP = 14;
if nVP > 1, rand('seed', nVP); end
basenames = {'online_noErrP_detection_', 'online_CalibErrP_detection_', 'online_OrigErrP_detection_'};
speller_calib_prefix = 'speller_calibration_';
ErrP_calib_prefix = 'ErrP_calibration_';
phrases = {'LET_YOUR_BRAIN_TALK', ...
           'MAY_THE_FORCE_BE_WITH_YOU', ...
           'ICH_BIN_EIN_BERLINER'};
%            'TALK_TO_THE_HAND', ...
block_idx = [1 2 1 2 2 3 1 3 3;
             1 1 2 2 3 1 3 2 3];
nBlocks = length(phrases) * 3;
phrases_idx = [randperm(nBlocks/3);randperm(nBlocks/3);randperm(nBlocks/3)];
phrases = phrases(phrases_idx(sub2ind([3 nBlocks/3], block_idx(1,:), block_idx(2,:))));


%% Start pyff
pyff('startup', 'a', [BCI_DIR 'python/pyff/src/Feedbacks/_VisualSpeller'], 'gui', 0);

%% Speller Calibration
run_online_speller_calibration

%% ErrP Calibration
run_online_ErrP_calibration

%% Train the classifiers
bbci= bbci_default;
bbci.train_file= {strcat(TODAY_DIR, speller_calib_prefix, VP_CODE), ...
                  strcat(TODAY_DIR, ErrP_calib_prefix, VP_CODE, '*')};
% bbci.train_file{1} = ...
% '/home/bbci/data/bbciRaw/VPiac_10_09_20/speller_calibration_VPiac';
% bbci.train_file{2} = ...
% '/home/bbci/data/bbciRaw/VPiac_10_09_20/ErrP_calibration_VPiac';
% bbci.save_name= strcat('/home/NicoSchmidt/data/Test');
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', basenames{2}, VP_CODE);
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type dbcont to continue\n');
keyboard
bbci_bet_finish
close all

%% Online Spelling **practice**
nr_sequences = bbci.setup_opts.nr_sequences;
offline_mode = 0;
do_ErrP_detection = 0;
desired_phrase = 'BCI';
fprintf('Press <RETURN> to start ErrP calibration practice.\n'),pause
log_filename = [TODAY_DIR basenames{1} 'practice_' VP_CODE '.log']; %#ok<*NASGU>

setup_online_speller
pyff('setdir','basename', [basenames{1} 'practice_']);
pyff('play');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller_ErrP_detection', 'bbci.fb_port', 12345);
pyff('stop');
pyff('quit');

%% Start Online Fixed Spelling Block 1-4:
nr_sequences = bbci.setup_opts.nr_sequences;
offline_mode = 0;
for iii=1:4
  fprintf('Press <RETURN> to start next block.\n'); pause;
  
  do_ErrP_detection = block_idx(1,iii)==2;
  basename = basenames{block_idx(1,iii)};
  desired_phrase = phrases{iii};
  if block_idx(2,iii)==1, log_filename = [TODAY_DIR basename VP_CODE '.log'];
  else log_filename = [TODAY_DIR basename VP_CODE '0' num2str(block_idx(2,iii)) '.log']; end

  setup_online_speller
  pyff('setdir', 'basename', basename);
  pyff('play'); pause(1)
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller_ErrP_detection', 'bbci.fb_port', 12345);
  pyff('stop');
  pyff('quit');
end


%% Train the ErrP classifier on online spelling data:
bbci_Calib = bbci; % save first classifier
bbci.func_mrk_opts.nRepetitions = bbci_Calib.setup_opts.nr_sequences;
bbci.train_file= {strcat(TODAY_DIR, basenames{1}, VP_CODE, '*'), ...
                  strcat(TODAY_DIR, basenames{2}, VP_CODE, '*')};
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', basenames{3}, VP_CODE);
% bbci.train_file{1} = ...
% '/home/bbci/data/bbciRaw/VPiac_10_09_20/online_noErrP_detection_VPiac*';
% bbci.train_file{2} = ...
% '/home/bbci/data/bbciRaw/VPiac_10_09_20/online_withErrP_detection_VPiac*';
bbci_bet_prepare
bbci.setup_opts.cfy_ival{2} = 'auto'; % set a few things from previous training
bbci.setup_opts.ErrP_bias = 0;
bbci.setup_opts.analyze_step = 5;
bbci_bet_analyze
fprintf('Type dbcont to continue\n');
keyboard
bbci_bet_finish
close all

% save classifier:
bbci_Orig = bbci;

%% Start Online Fixed Spelling Block 5-9:
for iii=5:9
  fprintf('Press <RETURN> to start next block.\n'); pause;
  
  do_ErrP_detection = block_idx(1,iii)~=1;
  basename = basenames{block_idx(1,iii)};
  desired_phrase = phrases{iii};
  if block_idx(2,iii)==1, log_filename = [TODAY_DIR basename VP_CODE '.log'];
  else log_filename = [TODAY_DIR basename VP_CODE '0' num2str(block_idx(2,iii)) '.log']; end
  
  if block_idx(1,iii)==3,
    bbci = bbci_Orig;
  else
    bbci = bbci_Calib;
  end

  setup_online_speller
  pyff('setdir', 'basename', basename);
  pyff('play'); pause(1)
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller_ErrP_detection', 'bbci.fb_port', 12345);
  pyff('stop');
  pyff('quit');
end
%%

% err_order = [0 1];
% if mod(nVP,2)==0, err_order = fliplr(err_order); end
% for iii=1:nBlocks/2
%   for do_ErrP_detection=err_order
%     if do_ErrP_detection
%       basename = basename_withErrP;
%     else
%       basename = basename_noErrP;
%     end
%     offline_mode = 0;
%     desired_phrase = phrases{(iii-1)*2 + do_ErrP_detection+1};
%     if iii==1, log_filename = [TODAY_DIR basename VP_CODE '.log'];
%     else       log_filename = [TODAY_DIR basename VP_CODE '0' num2str(iii) '.log']; end
% 
%     setup_online_speller
%     pyff('setdir', 'basename', basename);
%     pyff('play'); pause(1)
%     bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller_ErrP_detection', 'bbci.fb_port', 12345);
%     pyff('stop');
%     pyff('quit');
%     
%     fprintf('Press <RETURN> to start next block.\n'); pause;
%   end
% end
%%
fprintf('Finshed experiment. YOU''RE FREE!!! \t\t...for now...\n');
