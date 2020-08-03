%% P300 experiments in Leipzig

%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
%fprintf('Prepare cap. Press <RETURN> when finished.\n'), pause

%% Basic settings
COPYSPELLING_FINISHED = 246;
nVP = 14;

if nVP > 1, 
    rand('seed', nVP); 
end

basenames = {'online_noErrP_detection_', 'online_CalibErrP_detection_', 'online_OrigErrP_detection_'};

speller_calib_prefix = 'speller_calibration_';

phrases = {'LET_YOUR_BRAIN_TALK', ...
           'MAY_THE_FORCE_BE_WITH_YOU', ...
           'ICH_BIN_EIN_BERLINER'};

block_idx = [1 2 1 2 2 3 1 3 3;
             1 1 2 2 3 1 3 2 3];

nBlocks = length(phrases) * 3;
phrases_idx = [randperm(nBlocks/3);randperm(nBlocks/3);randperm(nBlocks/3)];
phrases = phrases(phrases_idx(sub2ind([3 nBlocks/3], block_idx(1,:), block_idx(2,:))));


%% Start pyff
%pyff('startup', 'a', ['D:\development\2011.5/src/Feedbacks/_VisualSpeller'], 'gui', 0);

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'dir', 'D:\development\2011.5\src', 'bvplugin',0 );

%% Speller Calibration
run_calibration


%% Train the classifiers
bbci= bbci_default;
bbci.train_file= {strcat(TODAY_DIR, speller_calib_prefix, VP_CODE), ...
                  strcat(TODAY_DIR, ErrP_calib_prefix, VP_CODE, '*')};

bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', basenames{1}, VP_CODE);

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
  
  if block_idx(2,iii)==1, 
      log_filename = [TODAY_DIR basename VP_CODE '.log'];
  else
      log_filename = [TODAY_DIR basename VP_CODE '0' num2str(block_idx(2,iii)) '.log'];
  end

  setup_online_speller

  pyff('setdir', 'basename', basename);
  pyff('play'); pause(1)
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller_ErrP_detection', 'bbci.fb_port', 12345);
  pyff('stop');
  pyff('quit');
  
end

fprintf('Finshed experiment. YOU''RE FREE!!! \t\t...for now...\n');
