subdir= 'BCIC_IVa';
%subdir= 'BCIC_IVb';
%subdir= 'BCIC_IVc';

bbci= [];
bbci.setup= 'cspauto';
bbci.train_file= strcat('/mnt/usb/data/bci_competition_iv/', subdir, ...
                        '/imag_arrow*');
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
bbci.classDef= {1, 2, 3; 'left', 'right', 'foot'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.setup_opts.usedPat= 'auto';

bbci_bet_prepare
bbci_bet_analyze





bbci.train_file= [EEG_RAW_DIR 'VPik_08_04_22/imag_arrow*'];

showGuiEEG(proc_selectChannels(Cnt, 'F3,z,4','C3,z,4','P3,z,4'), mrk)

