global DATA_DIR IO_ADDR IO_LIB
global TODAY_DIR VP_CODE VP_SCREEN
global ACQ_PREFIX_LETTER ACQ_LETTER_START

VP_SCREEN= [1680 0 1024 768];

ACQ_PREFIX_LETTER= 't';
ACQ_LETTER_START= 'c';

DATA_DIR= 'd:\data\';

parport= 'PC';
%parport= 'USB2LPT';
%parport= 'PCI2LPT';

% Specification of the addresse of the parallel port:
% Prüfen/umstellen: Systemsteuerung -> System -> Hardware -> Geräte-Manager
% -> Anschlüsse (COM und LPT): Double-Click -> Resourcen: 1. Adresse bei
% E/A-Bereich
switch(parport)
  case 'PC',
    IO_ADDR= hex2dec('DC00');
%     IO_ADDR= '127.0.0.1';
%    IO_ADDR= hex2dec('7800');
 % case 'Docking',
   % IO_ADDR= hex2dec('3BC');
 % case 'USB2LPT',
%    IO_ADDR= hex2dec('378');
  %  IO_ADDR= hex2dec('278');
  %case 'PCI2LPT',
  %  IO_ADDR= hex2dec('4CF8');
%    IO_ADDR= hex2dec('4008');
  otherwise,
    error('wrong setting for parport');
end
IO_LIB= which('inpout32.dll');
fprintf('Assuming Trigger cable is connected to %s. If not edit ''%s''.\n', parport, mfilename)
set_general_port_fields('localhost');
startup_bbci;