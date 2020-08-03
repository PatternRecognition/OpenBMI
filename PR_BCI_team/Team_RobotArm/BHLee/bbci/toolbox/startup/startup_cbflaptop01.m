global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_PREFIX_LETTER

%VP_SCREEN= [5 548 640 480];
%VP_SCREEN= [-959 0 960 600];
VP_SCREEN= [-1919 0 1920 1200];
ACQ_PREFIX_LETTER= 'v';

DATA_DIR= 'c:\data\';

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
REMOTE_RAW_DIR= EEG_RAW_DIR;
set_general_port_fields({'cbflaptop01','cbflaptop01'});
