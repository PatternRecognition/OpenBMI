startup_laptop_martijn;
setup_bbci_online;
set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'pyff';


file = 'd:/data/bbciRaw/VPlb_08_09_30/imag_fbarrowVPlb03';

setup = 'd:/data/bbciRaw/VPlb_08_09_30/tmp_classifier.mat';

Wps= [40 49]/1000*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
[filt.b, filt.a]= cheby2(n, 50, Ws);

[Cnt, mrk_orig] = eegfile_loadBV(file, 'fs', 100, 'filt', filt, 'clab', {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'});

% classDef= {1, 2, 3; 'left', 'right', 'foot'};
% mrk= mrk_defineClasses(mrk_orig, classDef);


bbci_bet_apply_offline(Cnt, mrk_orig, 'setup_list', setup, 'realtime', 1);

