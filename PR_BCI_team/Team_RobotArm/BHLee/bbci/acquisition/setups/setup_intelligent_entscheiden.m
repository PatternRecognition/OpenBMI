VP_SCREEN = [-1280 0 1280 1024];
VP_CODE = 'VPzq';
CLSTAG = 'LR';

if isempty(CLSTAG),
  error('Variable CLSTAG has to be defined');
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

setup_bbci_online; %% needed for acquire_bv

path([BCI_DIR 'acquisition/setups/intelligent_entscheiden'], path);

fprintf('\n\nWelcome to intelligent-entscheiden Live BCI Experiment\n\n');

bvr_sendcommand('stoprecording');

% bvr_sendcommand('loadworkspace', 'FastnEasy_motor');      %% Berlin's Fast'n'Easy Caps
% bvr_sendcommand('loadworkspace', 'one_channel'); 
bvr_sendcommand('loadworkspace', ['season10_64ch_FastnEasy']);
try
  bvr_checkparport('type','S');
catch
  error('BrainVision Recorder must be running.\nThen restart %s.', mfilename);
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];

[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci= [];
bbci.setup= 'cspauto';
bbci.train_file= strcat(subdir, '/imag_arrow*');
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
%bbci.classDef= {1, 2; 'left', 'foot'};
%bbci.classes= {'left', 'foot'};
bbci.classDef= {1, 2; 'left', 'right'};
bbci.classes= {'left', 'right'};
bbci.feedback= 'feedback_hexawrite';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
bbci.setup_opts.usedPat= 'auto';

%If'auto' mode does not work robustly:
%bbci.setup_opts.usedPat= [1:6];