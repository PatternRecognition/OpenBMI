global DATA_DIR
res_dir = [DATA_DIR 'results/csp_paramspace_online/'];
subject = 'Dummy';
date_str = '08_02_01';
class_tag = 'LR';

bbci.train_file= {};
% Note: The channel labels are acquired from BV recorder, therefore
%       it's required to let the BV recorder run during execution
%       of bbci_bet_prepare.
bbci.player = 1;
bbci.setup = 'adaptive'; 
bbci.save_name = [subject '_' date_str '/adaptive_' subject];
bbci.feedback = '1d';
bbci.classes = {'left','right'};
bbci.fs = 100;
bbci.host = 'tubbci.ml';
bbci.setup_opts= struct;
bbci.withclassification = false;
