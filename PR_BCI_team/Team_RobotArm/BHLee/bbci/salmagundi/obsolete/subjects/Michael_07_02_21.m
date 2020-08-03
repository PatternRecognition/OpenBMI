global DATA_DIR
res_dir = [DATA_DIR 'results/csp_paramspace_online/'];
subject = 'Michael';
date_str = '07_02_21';
class_tag = 'LR';

bbci.train_file= {};
% Note: The channel labels are acquired from BV recorder, therefore
%       it's required to let the BV recorder run during execution
%       of bbci_bet_prepare.
bbci.player = 1;
bbci.setup = 'cspproto'; 
bbci.save_name = [subject '_' date_str '/cspproto_' subject];
bbci.feedback = '1d';
bbci.classes = {'left','right'};
bbci.fs = 100;
bbci.epo_file = [res_dir 'epochs_' subject '_' class_tag];
bbci.host = 'bbcilab';
bbci.setup_opts= struct;
