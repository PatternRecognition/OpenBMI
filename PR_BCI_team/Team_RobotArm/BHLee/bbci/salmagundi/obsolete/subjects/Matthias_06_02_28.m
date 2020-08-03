bbci.train_file= strcat('Matthias_06_02_28/imag_lettMatthias', {'','2'});
bbci.classDef = {1,2;'left','right'};
bbci.player = 1;
bbci.setup = 'csp';
bbci.save_name = 'Matthias_06_02_28/imag_Matthias';
bbci.feedback = '1d';
bbci.classes = {'left','right'};

bbci.setup_opts= struct('band', [10 15], ...
                        'ival', [800 4000], ...
                        'nPat', 2, ...
                        'threshold', 140);
