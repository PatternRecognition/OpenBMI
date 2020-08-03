% In case the subject already has a subject code,
% define the VP_CODE accordingly, e.g., VP_CODE= 'VPzq'
% before executing this script!

setup_bbci_online;

global TODAY_DIR REMOTE_RAW_DIR
acq_getDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

bbci= [];
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'imag_arrow*');
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
bbci.classDef= {1, 2, 3; 'left', 'right', 'foot'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
bbci.setup_opts.usedPat= 'auto';
%If'auto' mode does not work robustly:
%bbci.setup_opts.usedPat= [1:6];
