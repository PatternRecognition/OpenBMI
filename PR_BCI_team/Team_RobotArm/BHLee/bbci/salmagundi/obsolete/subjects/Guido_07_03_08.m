global DATA_DIR
res_dir = [DATA_DIR 'results/csp_paramspace_online/'];
subject = 'Guido';
date_str = '07_03_08';
class_tag = 'RF';

bbci.train_file= {};
% Note: The channel labels are acquired from BV recorder, therefore
%       it's required to let the BV recorder run during execution
%       of bbci_bet_prepare.
bbci.player = 1;
bbci.feedback = '1d';
bbci.classes = {'foot','right'};
bbci.fs = 100;
bbci.epo_file = [res_dir 'epochs_' subject '_' class_tag];
bbci.host = 'bbcilab';
bbci.setup_opts= struct;
bbci.classDef = {-1, -2; 'foot','right'};

if cspproto
  bbci.save_name = [subject '_' date_str '/cspproto_' subject];
  bbci.setup = 'cspproto';
else
  bbci.train_file= {[subject '_' date_str '/imag_fb00' subject]};
  bbci.save_name = [subject '_' date_str '/csp_' subject];
  bbci.setup = 'csp';
  bbci.setup_opts.model = 'LSR';
end