global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE
global ACQ_LETTER_START ACQ_LETTER_RESERVED

ACQ_LETTER_START= 'i';
ACQ_LETTER_RESERVED= 'qtu';

DATA_DIR= 'd:\data\';
IO_ADDR= hex2dec('378');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
%setup_bbci_online;
IO_LIB = which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR= 'D:\data\bbciRaw\';
set_general_port_fields({'tubbci','tubbci'});
