session_name= 'labrotation2011_LianaKarapetyan';

addpath([BCI_DIR 'acquisition/setups/' session_name]);
fprintf('\n\nWelcome to the study "%s"!\n\n', session_name);

%try
%  checkparport('type','S');
%catch
%  error('Check amplifiers (all switched on?) and trigger cables.');
%end

global TODAY_DIR
acq_makeDataFolder;

%% Settings for online classification
% superfluous, when the new system is default
startup_new_bbci_online;

classDef= {[71:100], [31:60];
           'target', 'nontarget'};

BC.read_param= {'fs',100};
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {classDef};
BC.save.file= 'bbci_classifier_ERP_Speller';
BC.analyze_fcn= @bbci_calibrate_ERP_Speller_v0;
BC.settings.clab_erp= {'Cz','PO7'};
crit= strukt('maxmin', 120, ...
             'clab', '*', ...
             'ival', [100 800]);
BC.settings.reject_eyemovements= 1;
BC.settings.reject_eyemovements_crit= crit;

bbci= [];
bbci.calibrate= BC;

VP_SCREEN = [0 0 1280 1024];
fprintf(['Type run_''' session_name ''' and press <RET>\n']);
