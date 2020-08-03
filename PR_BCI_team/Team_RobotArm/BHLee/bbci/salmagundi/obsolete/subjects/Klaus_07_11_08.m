global DATA_DIR
res_dir = [DATA_DIR 'results/csp_paramspace_online/'];
subject = 'Klaus';
date_str = '07_11_08';
class_tag = 'LR';

bbci.train_file= {};
% Note: The channel labels are acquired from BV recorder, therefore
%       it's required to let the BV recorder run during execution
%       of bbci_bet_prepare.
bbci.player = 1;
bbci.feedback = '1d';
bbci.classes = {'left','right'};
bbci.fs = 100;
bbci.epo_file = [res_dir 'epochs_' subject '_' class_tag '_flipper'];
bbci.host = 'bbcilab';
bbci.setup_opts= struct;
bbci.setup_opts.nPat = 1;
bbci.classDef = {[1 -1], [2 -2]; 'left','right'};

if cspproto
  bbci.save_name = [subject '_' date_str '/cspproto_' subject];
  bbci.setup = 'cspproto';
  bbci.setup_opts.model = 'LSR';
else
  bbci.train_file= {[subject '_' date_str '/imag_fb00' subject '1']};
  bbci.save_name = [subject '_' date_str '/csp_' subject];
  bbci.setup = 'csp';
  bbci.setup_opts.model = 'LSR';
end