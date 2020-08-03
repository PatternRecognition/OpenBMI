addpath([BCI_DIR 'bbci_bet/feedbacks/back_versions/hexawrite_v0.3']);

opt_fb= struct('client_machines',[]);
opt_fb.position= [50 430 800 600];


global LOG_DIR

sub_dir= 'Matthias_06_02_22';
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];

logno= 10; start= 44.1;
replay('hexawrite', logno, 'start',start, 'opt_fb',opt_fb, ...
       'save',sprintf('hexawrite_%03d_%s', logno, sub_dir);


sub_dir= 'Guido_06_02_28';
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];

logno= 130; start= 0; stop=148;
replay('hexawrite', logno, 'start',start, 'stop',stop, 'opt_fb',opt_fb, ...
       'save',sprintf('hexawrite_%03d_%s', logno, sub_dir);


return

sub_dir= 'Guido_06_02_22';
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];

%% viel verpasst
logno= 21; start= 44.1;
replay('hexawrite', logno, 'start',start, 'opt_fb',opt_fb, ...
       'save',sprintf('hexawrite_%03d_%s', logno, sub_dir);

%% viel verpasst
logno= 22; start= 3; stop= % muesste gesucht werden
replay('hexawrite', logno, 'start',start, 'opt_fb',opt_fb, ...
       'save',sprintf('hexawrite_%03d_%s', logno, sub_dir);

