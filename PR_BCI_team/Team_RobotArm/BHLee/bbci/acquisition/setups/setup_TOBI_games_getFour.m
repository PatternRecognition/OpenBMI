% path([BCI_DIR 'acquisition/setups/vitalbci_season1'], path);
% startup_bbcilaptop;
acquire_func = @acquire_sigserv;
setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to TOBI Entertainment: get4 \n\n');
global TODAY_DIR REMOTE_RAW_DIR VP_CODE DATA_DIR EEG_RAW_DIR
VP_CODE = 'VPtest'
acq_makeDataFolder('log_dir',1);

[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci= [];
bbci.setup= 'get4';
bbci.train_file= strcat(subdir, '/get4_train_',VP_CODE, '*');
bbci.clab = {'*'};
% bbci.func_mrk = 'mrkodef_photobrowser_oddball';
bbci.classDef = {120,20;'Target', 'Non-target'}
% bbci.classes= 'auto';
bbci.feedback= '1d_AEP';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
% bbci.setup_opts.usedPat= 'auto';
%If'auto' mode does not work robustly:
%bbci.setup_opts.usedPat= [1:6];
bbci.fs = 100;
bbci.fb_machine = general_port_fields(1).bvmachine; 
bbci.fb_port = 12345;

% hdr = eegfile_readBVheader(bbci.train_file(1:end-1));
% bbci.filt.b = [];bbci.filt.a = [];
% Wps= [40 49]/hdr.fs*2;
% [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
% [bbci.filt.b, bbci.filt.a]= cheby2(n, 50, Ws);

addpath('D:\svn\bbci\acquisition\setups\TOBI_games_getFour')

bbci.withclassification = 1;
bbci.withgraphics = 1;