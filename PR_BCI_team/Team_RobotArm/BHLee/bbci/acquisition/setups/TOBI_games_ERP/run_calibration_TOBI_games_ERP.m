% send_xmlcmd_udp('init', general_port_fields.bvmachine, bbci.fb_port);

%% RUN CALIBRATION SESSION

% system(['cmd /C "' bbci.TOBI_GAMES_DIR(1:2) '& cd ' bbci.TOBI_GAMES_DIR(3:end) 'chess & run.bat "& '])
% system([bbci.TOBI_GAMES_DIR 'getFour\run.bat'])

sprintf('Please start the Game ''%s'' \n', bbci.GAME)
inp='';
while ~strcmp(inp, 'start')
inp = input('type ''start'' just before you start the calibration: ', 's');
end

[dum fname] = fileparts(bbci.train_file);
disp('recording is started!')
signalServer_startrecoding(fname)

disp('recording was stopped !')


%% analzye data
if ~strcmp(bbci.train_file(end), '*')
    bbci.train_file = [bbci.train_file '*'];
end
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;
close all; clear data;