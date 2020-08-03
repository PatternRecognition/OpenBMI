fprintf('\n\nWelcome to TOBI Entertainment: games \n\n');
global TODAY_DIR REMOTE_RAW_DIR VP_CODE DATA_DIR EEG_RAW_DIR acquire_func

% Wellcome to the world of the Signalserver and Gtec
acquire_func = @acquire_sigserv;
% acquire_func = @acquire_bv; %FOR TESTING!

setup_bbci_online; %% needed for acquire_bv

% VP_CODE = 'VPkam' %FOR testing
if isempty(VP_CODE),
    warning('VP_CODE undefined - assuming fresh subject  -- press Ctr-C to stop!')
    pause(5)
end

%  TODAY_DIR = 'D:/data/bbciRaw/VPkam_11_07_22/' %Ingrid testing connect4; %for testing
%  TODAY_DIR = 'D:\data\bbciRaw\VPlh_11_07_29/'
acq_makeDataFolder('log_dir',1, 'multiple_folders',1);

bbci= [];
%FOR TESTING
% TODAY_DIR = 'D:/data/bbciRaw/VPzq_11_03_22/' %Michael doing MI-connect4
% TODAY_DIR = 'D:/data/bbciRaw/VPfb_11_02_24/';
%TODAY_DIR = 'D:/data/bbciRaw/VPfb_11_03_03/'; % Jansen
% TODAY_DIR = 'D:/data/bbciRaw/VPfbc_11_03_04/' %Fissler


addpath([BCI_DIR 'acquisition\setups\TOBI_games_MI'])

[dmy, dmy2]= fileparts(TODAY_DIR(1:end-1));
bbci.subdir = [dmy2 '/'];

bbci.setup= 'csp_LRP';
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10'};
bbci.classes=  'auto';
bbci.classDef= {1, 2, 3; 'left','right','foot'};
bbci.feedback= '';
bbci.setup_opts.ilen_apply= 750;
bbci.adaptation.UC= 0.03;
bbci.adaptation.UC_mean= 0.075;
bbci.adaptation.UC_pcov= 0.03;
bbci.adaptation.load_tmp_classifier= 1; % CARMEN
bbci.start_marker = 210;
bbci.quit_marker = 254;

bbci.TOBI_GAMES_DIR = 'E:\temp\games\games\';


bbci.fs = 100;
set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'tobi_c';
bbci.fb_machine = '127.0.0.1';
bbci.fb_port = 12345;

bbci.fileNames ={};
bbci.fileNames{1}= ['imag_calib_bar_getFour_' VP_CODE];
bbci.fileNames{2}= ['imag_fb_bar_pcovmean_getFour_' VP_CODE];
bbci.fileNames{3}= ['imag_fb_bar_pmean_getFour_' VP_CODE];



bbci.classyNames= {};
bbci.classyNames{1} = ['bbci_classifier_' bbci.setup '_16chans_pcovmean_' VP_CODE];
bbci.classyNames{2} = ['bbci_classifier_' bbci.setup '_16chans_pmean_' VP_CODE];


%FOR TESTING
% bbci.fileNames{1}= ['imag_fbarrow_pretrained_ERD_24chan_mcc_'];
% bbci.fileNames{1}= ['imag_fbarrow_pretrained_ERD_24chan_mcc_']; %Jansen
% bbci.fileNames{1}= ['imag_fbarrow_LRPprecalc_']; %Fissler
% bbci.classyNames{1} = ['bbci_classifier_' bbci.setup '_24chans_pcovmean_' VP_CODE];

VP_SCREEN= [-1279 0 1280 1024];

bbci.tobi_c_classifier = { ...
    {'ERD' 'BBCI Motor Imagery ERD Classifier', 'dist', 'biosig', '0x300'},...
    {'LRP' 'BBCI Motor Imagery LRP Classifier', 'dist', 'biosig', '0x300'},...
    {'fuse' 'BBCI Motor Imagery fused Classifier', 'dist', 'biosig', '0x300'}...
    };

% send_tobi_c_udp('init', bbci.fb_machine, bbci.fb_port, bbci.tobi_c_classifier)
% disp('TOBI Interface C was set up with 3 classifier outputs!')

% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;


fprintf('Type ''edit run_tobi_games_MI'' and press <RET>.\n');



%%
