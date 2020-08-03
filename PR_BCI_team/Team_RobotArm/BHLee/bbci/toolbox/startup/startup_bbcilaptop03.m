global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global TMP_DIR LOG_DIR
global ACQ_PREFIX_LETTER

VP_SCREEN= [-1279 0 1280 1024];
%VP_SCREEN= [5 548 640 480];
ACQ_PREFIX_LETTER= 'b';

DATA_DIR= 'c:\data\';
TMP_DIR= 'd:\tmp\';
LOG_DIR= 'd:\tmp\';

parport= 'Docking';
%parport= 'USB2LPT';
switch(parport)
  case 'USB2LPT',
    IO_ADDR= hex2dec('378');
  case 'Docking',
    IO_ADDR= hex2dec('3BC');
  otherwise,
    error('wrong setting for parport');
end
fprintf('Assuming Trigger cable is connected to %s. If not edit ''%s''.\n', parport, mfilename)

startup_bbci;
setup_bbci_online;
IO_LIB= which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR = 'c:\data\bbciRaw\';
set_general_port_fields('hostbyname');
%set_general_port_fields('bbcilaptop03')
