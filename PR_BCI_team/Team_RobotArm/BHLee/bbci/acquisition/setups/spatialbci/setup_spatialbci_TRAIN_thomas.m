clear opt;

opt.speakerSelected = [6 2 4 1 5 3];

opt.language = 'english';

setup_spatialbci_GLOBAL

opt.filename = 'OnlineTrainFile';

opt.isi = 1000; % inter stimulus interval
opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'fixed';
opt.mode = 'copy';
opt.application = 'TRAIN_thomas';

opt.countdown = 0;
opt.repeatTarget = 3;
addpath([BCI_DIR 'acquisition/stimulation/spatial_auditory_p300/TRAIN_thomas']);