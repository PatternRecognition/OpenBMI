global DATA_DIR IO_ADDR IO_LIB TMP_DIR
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED

VP_SCREEN= [-1279 0 1280 1024];
ACQ_LETTER_START= 't';
ACQ_LETTER_RESERVED= 'qyz';

DATA_DIR= 'f:\Raw_Data\';
TMP_DIR= 'f:\tmp\';
IO_ADDR= hex2dec('CC00');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
%setup_bbci_online;
IO_LIB = which('inpout32.dll');

REMOTE_RAW_DIR= [DATA_DIR 'bbciRaw\'];
set_general_port_fields({'bbcipc','bbcipc'});
