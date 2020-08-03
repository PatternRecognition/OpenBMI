global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_PREFIX_LETTER PYFF_DIR
global BBCI_PRINTER 

BBCI_PRINTER = 1;
VP_SCREEN= [5 548 640 480];
ACQ_PREFIX_LETTER= 'e';

EEG_FIG_DIR = [TEX_DIR 'figures/'];
PYFF_DIR= 'c:\svn\pyff\src\';
DATA_DIR= 'c:\data\';
parport= 'Docking';
%parport= 'USB2LPT';
switch(parport)
  case 'USB2LPT',
    IO_ADDR= hex2dec('378');
  case 'Docking',
    IO_ADDR= hex2dec('3BC');
  otherwise,
    error('wrong setting for parport');
end
fprintf('Assuming Trigger cable is connected to %s. If not edit ''%s''.\n', parport, mfilename)

startup_bbci;
setup_bbci_online;
IO_LIB= which('inpout32.dll');

% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

overloader_dir= 'C:\toolbox_overloader';
dd= dir(overloader_dir);
if length(dd)>2,  % the first two are '.' and '..'
  warning('The following functions are overloaded:');
  dir(overloader_dir)
  addpath(overloader_dir);
end

% Run everything on this machine.
REMOTE_RAW_DIR = EEG_RAW_DIR;
set_general_port_fields('hostbyname');
