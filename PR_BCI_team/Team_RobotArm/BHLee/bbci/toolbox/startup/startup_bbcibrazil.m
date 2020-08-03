global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED
global TMP_DIR LOG_DIR

% VP_SCREEN= [1285 5 1280 1024];
% % VP_SCREEN= [5 548 640 480];
% % VP_SCREEN= [-1919 0 1920 1200];
ACQ_LETTER_START= 'braz';
ACQ_LETTER_RESERVED= 'il';


DATA_DIR= 'D:/data/';
TMP_DIR= DATA_DIR;
LOG_DIR= DATA_DIR;
% IO_ADDR= hex2dec('3BC');
IO_ADDR= hex2dec('E020');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
setup_bbci_online;
IO_LIB= which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR = 'D:\data\bbciRaw';
set_general_port_fields('hostbyname');

disp('If you want a specific patient''s code, please type: ''VP_CODE=xx'' otherwise a new code will be created.');
disp('Type "setup_Brazil" and press <RETURN> to setup the BCI paradigm');