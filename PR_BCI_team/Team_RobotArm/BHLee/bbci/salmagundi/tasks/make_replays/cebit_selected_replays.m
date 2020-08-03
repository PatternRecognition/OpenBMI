setup_bbci_bet
unix('xset s off');
unix('xset -dpms');

opt_fb= struct('client_machines',[]);
opt_fb.position= [1 410 800 600];
opt_fb.text_spec= {'FontName','Courier'};
opt_movie= struct('freeze_out', 2);
opt_movie.fade_out= 1;
opt_movie.overwrite= 1;
opt= strukt('opt_fb',opt_fb, 'opt_movie',opt_movie);
opt.position= [1 410 800 600];
opt.force_set= {{63, 'FontName','Courier', 'FontSize',0.05}};

global LOG_DIR


sub_dir= 'Guido_06_03_09'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];


logno= 180; opt.start= 99; opt.stop= 338;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hex-o-spell_%03d', sub_dir, logno));
%% BERLIN_BRAIN_COMPUTER_INTERFACE. 32 chars in 231 s: 8.3 char/min

logno= 154; opt.start= 68; opt.stop= 278;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hex-o-spell_%03d', sub_dir, logno));
%% DIE_SONNE_IST_VON_KUPFER. 25 chars in 203 s: 7.4 char/min

logno= 177;  opt.start= 1215; opt.stop= 1454;
replay('hexawrite', logno, opt, ...
       'save',sprintf('%s_hex-o-spell_%03d', sub_dir, logno));
%% BERLIN_BRAIN_COMPUTER_INTERFACE. 32 chars in 232 s: 8.3 char/min
