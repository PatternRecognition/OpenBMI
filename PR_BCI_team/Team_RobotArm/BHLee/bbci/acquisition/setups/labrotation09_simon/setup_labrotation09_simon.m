path([BCI_DIR 'acquisition/setups/labrotation09_simon'], path);
addpath D:\svn\bbci\investigation\personal\simooon\pci

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome!\n\n');

%If you like to crash the whole computer, do this:
%system('c:\Vision\Recorder\Recorder.exe &')

%If matlab crashed before, BVR might still be in recording mode
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu_EMG2'); 

try
  bvr_checkparport('type','S');
catch
  error('BrainVision Recorder must be running.\nThen restart %s.', mfilename);
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

overloader_dir= 'D:\svn_overloader\labrotation09_simon\';
dd= dir([overloader_dir '/*.m']);
if ~isempty(dd),
  addpath(overloader_dir);
  fprintf('The following functions have been overloader:\n');
  fprintf('  %s\n', dd.name);
end
clear dd overloader_dir
