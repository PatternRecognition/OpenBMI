global DATA_DIR IO_ADDR IO_LIB TMP_DIR PYFF_DIR
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED

VP_SCREEN= [-1919 0 1920 1200];
ACQ_LETTER_START= 'i';
ACQ_LETTER_RESERVED= 'qtu';

DATA_DIR= 'd:\data\';
TMP_DIR= 'd:\temp\';
PYFF_DIR= 'd:\svn\pyff\src\';
IO_ADDR= hex2dec('2030');
%This is for USB2LTP:
% IO_ADDR= hex2dec('378');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
setup_bbci_online;
IO_LIB = which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR= [DATA_DIR 'bbciRaw\'];
set_general_port_fields({'tubbci2','tubbci2'});
%set_general_port_fields('localhost');

VP_SCREEN= [-1919 0 1920 1200];

cd(BCI_DIR);
