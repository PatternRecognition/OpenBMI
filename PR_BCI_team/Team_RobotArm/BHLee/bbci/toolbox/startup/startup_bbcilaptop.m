global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED
global TMP_DIR LOG_DIR

VP_SCREEN= [-1919 0 1920 1200];
ACQ_LETTER_START= 'i';
ACQ_LETTER_RESERVED= 'qtu';
TMP_DIR= 'c:\temp\';
LOG_DIR= 'c:\temp\';

DATA_DIR= 'c:\data\';
IO_ADDR= hex2dec('3BC');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
setup_bbci_online;
IO_LIB= which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR = 'c:\data\bbciRaw\';
set_general_port_fields('hostbyname');
