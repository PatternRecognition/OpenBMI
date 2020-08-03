global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_PREFIX_LETTER

VP_SCREEN= [1366 0 1920 1200];
ACQ_PREFIX_LETTER= 'l';

DATA_DIR= 'd:\data\';

%parport= 'USB2LPT';
parport= 'PCI2LPT';

switch(parport)
  case 'USB2LPT',
    %IO_ADDR= hex2dec('378');
    IO_ADDR= hex2dec('278');
  case 'PCI2LPT',
% prüfen/umstellen: Systemsteuerung -> System -> Hardware -> Geräte-Manager
% -> Anschlüsse (COM und LPT): Double-Click -> Resourcen: 1. Adresse bei
% E/A-Bereich  
    %IO_ADDR= hex2dec('4CF8'); 
   IO_ADDR= hex2dec('4008');
  otherwise,
    error('wrong setting for parport');
end
fprintf('Assuming Trigger cable is connected to %s. If not edit ''%s''.\n', parport, mfilename)

startup_bbci;
setup_bbci_online;
IO_LIB= which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR = EEG_RAW_DIR;
set_general_port_fields('localhost');
