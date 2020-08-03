global DATA_DIR IO_ADDR IO_LIB TMP_DIR
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_LETTER_START ACQ_LETTER_RESERVED ACQ_PREFIX_LETTER

% VP_SCREEN= [-1919 0 1920 1200];
%samsug monitor psylab
VP_SCREEN=[-1279 0 1280 1024];
ACQ_PREFIX_LETTER= 'g';

DATA_DIR= 'd:\data\';
TMP_DIR= 'd:\temp\';
IO_ADDR= hex2dec('5C00');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

startup_bbci;
setup_bbci_online;
IO_LIB = which('inpout32.dll');

% overloader_dir= 'D:\toolbox_overloader';
% dd= dir(overloader_dir);
% if length(dd)>2,  % the first two are '.' and '..'
%   warning('The following functions are overloaded:');
%   dir(overloader_dir)
%   addpath(overloader_dir);
% end

% Run everything on this machine.
REMOTE_RAW_DIR= [DATA_DIR 'bbciRaw\'];
%set_general_port_fields({'user-fc1f6c5865','user-fc1f6c5865'});
set_general_port_fields('localhost');