fprintf('\n\nWelcome to TOBI Entertainment: games \n\n');
global TODAY_DIR REMOTE_RAW_DIR VP_CODE DATA_DIR EEG_RAW_DIR acquire_func
start_signalserver('server_config_visualERP.xml') 

input('please DEBLOCK the parallel port!');
input('DID YOU REALLY, REALLY DEBLOCK THE PARALLEL PORT?');

% Wellcome to the world of the Signalserver and Gtec 
acquire_func = @acquire_sigserv;
setup_bbci_online; %% needed for acquire_bv


if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject  -- press Ctr-C to stop!')
  pause(5)
end

acq_makeDataFolder('log_dir',1, 'multiple_folders',1);

bbci= [];
game = input('Please specify the game you want to play: ''getFour'' or ''javaChess''\n', 's');
switch game
    case 'getFour'
        bbci.classDef = {[120:126], [20:26]; 'Target', 'Non-target'};
    case 'javaChess'
        bbci.classDef = {[120:140], [20:40]; 'Target', 'Non-target'};
    otherwise
        error('game not specified correctly. Only getFour and javaChess are implemented!')
end
addpath([BCI_DIR 'acquisition\setups\TOBI_games_ERP'])

[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
%bbci.setup= 'PhotoBrowser';
bbci.setup= 'TOBI_games_ERP';
bbci.train_file = strcat(subdir, '/', game, '_train_',VP_CODE);
bbci.online_file = strcat(subdir, '/', game, '_online_',VP_CODE);
bbci.clab = {'*'};

bbci.feedback= '1d_AEP';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
bbci.fs = 100;
set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'tobi_c';
bbci.fb_machine = '127.0.0.1'; 
% bbci.fb_port = 12349;
bbci.fb_port = 12345;

bbci.TOBI_GAMES_DIR = 'E:\temp\games\games\';
bbci.GAME = game;

bbci.filt.b = [];bbci.filt.a = [];
Wps= [40 49]/1200*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
[bbci.filt.b, bbci.filt.a]= cheby2(n, 50, Ws);

bbci.withclassification = 1;
bbci.withgraphics = 1;


disp('to start calibration, enter:       run_calibration_TOBI_games_ERP')
disp('to start online-gaming, enter:     run_online_TOBI_games_ERP')        
