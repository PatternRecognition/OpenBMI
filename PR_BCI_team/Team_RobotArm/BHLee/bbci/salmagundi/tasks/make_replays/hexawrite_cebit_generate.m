setup_bbci_bet
global LOG_DIR

opt= struct('verbose',1);
opt.fid= fopen([BCI_DIR 'tasks/make_replays/cebit_contents.txt'], 'wt');


sub_dir= 'Guido_06_03_09'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];
log_contents_hexawrite([], opt);

sub_dir= 'Guido_06_03_10'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];
log_contents_hexawrite([], opt);

sub_dir= 'Michael_06_03_09'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];
log_contents_hexawrite([], opt);

sub_dir= 'Michael_06_03_10'
LOG_DIR= ['/home/neuro/data/BCI/bbciRaw/' sub_dir '/log/'];
log_contents_hexawrite([], opt);

fclose(opt.fid);
