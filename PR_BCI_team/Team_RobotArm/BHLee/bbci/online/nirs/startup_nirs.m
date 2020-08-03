global DATA_DIR IO_ADDR IO_LIB TMP_DIR PYFF_DIR
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED
global general_port_fields

VP_SCREEN= [-1919 0 1920 1200];
ACQ_LETTER_START= 'i';
ACQ_LETTER_RESERVED= 'qtu';

currDir = cd;

DATA_DIR= ['D:\data\nirs\'];
TMP_DIR= [currDir '\temp\'];
PYFF_DIR= [currDir '\ToolBoxBCI\python\pyff\src'];
IO_ADDR= hex2dec('2030');
%This is for USB2LTP:
% IO_ADDR= hex2dec('378');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci_nirs;
setup_bbci_online;
IO_LIB = which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR= [DATA_DIR 'uni\'];
set_general_port_fields({'nirs','nirs'});
set_general_port_fields('localhost');

VP_SCREEN= [-1919 0 1920 1200];

%cd(BCI_DIR);
