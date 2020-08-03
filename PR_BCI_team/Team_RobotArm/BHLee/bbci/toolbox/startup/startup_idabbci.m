global IDABOX_DIR DATA_DIR IO_ADDR IO_LIB
global REMOTE_RAW_DIR VP_CODE
global ACQ_LETTER_START ACQ_LETTER_RESERVED

ACQ_LETTER_START= 'i';
ACQ_LETTER_RESERVED= 'q';

IDABOX_DIR= 'd:\neuro_toolbox\matlab\';
%DATA_DIR= 'd:\data\';
IO_ADDR= hex2dec('378');

% avoid "assert overload" conflict:
warning('off','MATLAB:dispatcher:nameConflict');
% avoid annoying behaviour of set_defaults:
warning('off','set_defaults:DEFAULT_FLD');

cd('d:\neuro_cvs\matlab\bci');
DATA_DIR='d:\data\';
startup_bci;
IO_LIB= which('inpout32.dll');
DATA_DIR= [BCI_DIR 'data/'];

% Run everything on this machine.
REMOTE_RAW_DIR = 'd:\data\bbciRaw\';
%set_general_port_fields('localhost');
BV_MACHINE= 'bbcilab';
APPLY_MACHINE= 'bbcilab';
general_port_fields= strukt('bvmachine',BV_MACHINE,...
                            'control',{APPLY_MACHINE,12470,12489},...
                            'graphic',{[],12471});

tol= ['d:\matlab\bbci_toolbox_overloader'];
dd= dir(tol);
dd(1:2)= [];
if ~isempty(dd),
  fprintf('\nWarning: The following BBCI toolbox functions are overloaded:\n');
  fprintf('  %s\n', dd.name);
  fprintf('\n');
  addpath(tol);
end

%% ?? New ppTrigger of Max does not work ??
addpath('d:\matlab\parallelport');
