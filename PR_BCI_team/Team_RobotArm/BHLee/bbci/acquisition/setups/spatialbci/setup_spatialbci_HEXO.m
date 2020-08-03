clear opt;

opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];
opt.language = 'german';

setup_spatialbci_GLOBAL

opt.filename = 'OnlineRunFile';

opt.isi = 175; % inter stimulus interval
opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'fixed';
opt.mode = 'free';
opt.application='HEXO_spell';
opt.maxRounds = 15;