calib= {'lett','lett2','move','move2'};
bbci.train_file= strcat('Maki_06_07_07/imag_', calib, 'Maki');
bbci.classDef = {1,2,3;'left','right','foot'};
bbci.player = 1;
bbci.setup = 'csp';
bbci.save_name = 'Maki_06_07_07/imag_Maki';
bbci.feedback = '1d';
bbci.classes = {'left','right'};
