global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR REMOTE_RAW_DIR VP_CODE VP_SCREEN
global ACQ_PREFIX_LETTER

ACQ_PREFIX_LETTER= 'd';
screen_pos= get(0, 'ScreenSize');
%VP_SCREEN= [-1919 0 1920 1200];
VP_SCREEN= [screen_pos(3) 0 1920 1200];

DATA_DIR= 'd:/data/';

parport= 'Docking';
% parport= 'USB2LPT';
%parport= 'PCI2LPT';

switch(parport)
  case 'USB2LPT',
%     IO_ADDR= hex2dec('378'); % prüfen ob das funktioniert --> gerätemanager --> parallelport: welche Adresse?
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
IO_LIB= which('inpout32.dll');

clear screen_pos parport
