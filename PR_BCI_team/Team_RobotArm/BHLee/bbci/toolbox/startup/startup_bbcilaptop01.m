global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_PREFIX_LETTER

%VP_SCREEN= [5 548 640 480];
%VP_SCREEN= [-959 0 960 600];
VP_SCREEN= [-1919 0 1920 1200];
ACQ_PREFIX_LETTER= 'b';

DATA_DIR= 'd:/data/';

parport= 'Docking';
%parport= 'USB2LPT';
%parport= 'PCI2LPT';

switch(parport)
  case 'USB2LPT',
    %IO_ADDR= hex2dec('378'); % prüfen ob das funktioniert --> gerätemanager --> parallelport: welche Adresse?
    IO_ADDR= hex2dec('278'); % prüfen ob das funktioniert --> gerätemanager --> parallelport: welche Adresse?
  case 'Docking',
    IO_ADDR= hex2dec('3BC');
  case 'PCI2LPT',
    IO_ADDR= hex2dec('FEE8');
  otherwise,
    error('wrong setting for parport');
end
fprintf('Assuming Trigger cable is connected to %s. If not edit ''%s''.\n', parport, mfilename)

startup_bbci;
setup_bbci_online;
IO_LIB= which('inpout32.dll');

% Run everything on this machine.
REMOTE_RAW_DIR = EEG_RAW_DIR;
set_general_port_fields('localhost');   % otherwise pyff does not receive control signals from bbci_bet_apply
