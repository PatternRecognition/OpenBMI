startup_bbci;
setup_bbci_online;
global TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE;

set_general_port_fields('localhost')
general_port_fields.feedback_receiver = 'pyff';

acq_getDataFolder('multiple_folders',1);
% subdir= 'VPlb_08_09_30';
% VP_CODE = subdir(1:4);

bbci_setup_cspauto;

bbci= [];
bbci.fb_machine = general_port_fields(1).bvmachine; 
bbci.feedback_receiver = 'pyff';
bbci.fb_port = 12345;
bbci.control_port = 33333;
% bbci.control_port = general_port_fields(1).control;
bbci.setup= 'cspauto_multitime';
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.train_file= strcat(subdir, '/imag_fbarrow_LapC3z4_LR', VP_CODE, '*');
% bbci.artifacts_file= strcat(subdir, '/artifacts', VP_CODE, '*');
% bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
bbci.clab = {'not', 'Fp*'};
bbci.classDef= {1, 2, 3; 'left', 'right', 'foot'};
bbci.classes= 'auto';
bbci.feedback= 'multi';
bbci.cmb_setup = 'concat';
bbci.save_name= [TODAY_DIR '/tmp_classifier'];
bbci.setup_opts.usedPat= 'auto';

Wps= [40 49]/1000*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
[bbci.filt.b, bbci.filt.a]= cheby2(n, 50, Ws);

bbci= set_defaults(bbci,'host', 'localhost', ...
                        'filt', [], ...
                        'clab', '*', ...
                        'impedance_threshold', 50, ...
                        'fs', 100);
                    
bbci_bet_memo_opt = struct;
bbci.setup_opts.usedPat= 'auto';             

bbci.withclassification = 1;
bbci.withgraphics = 1;