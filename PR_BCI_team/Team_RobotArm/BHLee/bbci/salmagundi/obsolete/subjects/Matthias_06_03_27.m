bbci.train_file= 'Guido_06_03_27/imag_lettGuido';
bbci.classDef = {1,2;'left','right'};
bbci.player = 2;
bbci.setup = 'csp';
bbci.save_name = 'Matthias_06_03_27/imag_Matthias';
bbci.feedback = '1d';
bbci.classes = {'left','right'};

bbci.setup_opts= struct('band', [10 15], ...
                        'ival', [800 4000], ...
                        'nPat', 2, ...
                        'threshold', 140);
