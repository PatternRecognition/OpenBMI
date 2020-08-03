global DATA_DIR IO_ADDR IO_LIB TMP_DIR
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED ACQ_PREFIX_LETTER

VP_SCREEN= [-1919 0 1920 1200];
ACQ_PREFIX_LETTER= 'n';

DATA_DIR= 'e:\data\';
TMP_DIR= [DATA_DIR 'tmp\'];
IO_ADDR= hex2dec('0378');
IO_LIB = which('inpoutx64.dll');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
startup_new_bbci_online;
