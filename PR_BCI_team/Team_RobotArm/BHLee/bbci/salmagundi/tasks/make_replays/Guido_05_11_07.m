setup_bbci_bet_unstable
unix('xset s off');
unix('xset -dpms');

opt_fb= struct('client_machines',[]);
opt_fb.position= [1 410 800 600];
opt_movie= struct('freeze_out', 2);
opt_movie.fade_out= 1;
opt_movie.overwrite= 1;
opt= strukt('opt_fb',opt_fb, 'opt_movie',opt_movie);
opt.position= [1 410 800 600];
opt.force_set= {{3, 'FontName','Helvetica', 'FontWeight','bold', 'FontSize',65}, ...
                {4, 'FontName','Helvetica', 'FontWeight','bold'}, ...
                {5, 'FontName','Helvetica', 'FontWeight','bold'}};

global LOG_DIR


sub_dir= 'Guido_05_11_07'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];

opt.start= 1.9;
opt.stop= 81;
replay('2d',13, opt, 'save','Guido_05_11_07_cursor_013');

