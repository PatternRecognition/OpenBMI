clear bbci

bbci.train_file= {'Matthias_06_02_22/imag_lettMatthias'};
bbci.classDef = {1,2,3;'left','right','foot'};
bbci.player = 1;
bbci.setup = 'bandpowerOnRaw';
bbci.save_name = 'Matthias_06_02_22/imag_Matthias';
bbci.feedback = '1d';
bbci.classes = {'left','right'};
bbci.fs= 1000;



return;
bbci.setup_opts.clab= {'*'};
bbci.setup_opts.ival= [];
bbci.setup_opts.default_ival= [750 3500];
bbci.setup_opts.band= [];
bbci.setup_opts.filtOrder= 5;
bbci.setup_opts.ilen_apply= 1000;
bbci.setup_opts.dar_ival= [-500 5000];
bbci.setup_opts.model= 'LSR';
bbci.setup_opts.threshold= inf;

