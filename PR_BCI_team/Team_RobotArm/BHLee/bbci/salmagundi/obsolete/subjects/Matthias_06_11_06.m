global DATA_DIR
res_dir = [DATA_DIR 'results/csp_paramspace_online/'];
subject = 'Matthias';
class_tag = 'LR';

bbci.train_file= {'Matthias_06_02_09/arteMatthias'};
% this is required to get matching channel labels!
bbci.classDef = {0;''};
bbci.player = 1;
bbci.setup = 'cspproto';
bbci.save_name = 'Matthias_06_11_06/cspproto_Matthias';
bbci.feedback = '1d';
bbci.classes = {'left','right'};
bbci.fs = 100;
bbci.epo_file = [res_dir 'epochs_' subject '_' class_tag];

bbci.setup_opts= struct;
