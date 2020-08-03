global DATA_DIR IO_ADDR IO_LIB TMP_DIR
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED ACQ_PREFIX_LETTER

%VP_SCREEN= [-1919 0 1920 1200];
VP_SCREEN= [1921 0 1920 1200];
ACQ_PREFIX_LETTER= 'j';

DATA_DIR= 'e:\data\';
TMP_DIR= 'e:\tmp\';
IO_ADDR= hex2dec('7800');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
setup_bbci_online;
IO_LIB = which('inpoutx64.dll');

% Run everything on this machine.
REMOTE_RAW_DIR= [DATA_DIR 'bbciRaw\'];
%set_general_port_fields({'tubbci6','tubbci6'});
set_general_port_fields('localhost');
