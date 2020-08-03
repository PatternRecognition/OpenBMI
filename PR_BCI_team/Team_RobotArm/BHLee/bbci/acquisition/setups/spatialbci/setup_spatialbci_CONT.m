clear opt;

opt.toneDuration = 40;
opt.speakerSelected = [6 2 4 1 5 3];

setup_spatialbci_GLOBAL

opt.filename = 'OnlineCONTFile';

opt.isi = 175; % inter stimulus interval
opt.isi_jitter = 0; % defines jitter in ISI

opt.itType = 'continuous';
opt.mode = 'free';
opt.application='CONT_navi';
opt.contMemory = 2;
opt.dTime = 'trialwise';
opt.degrees = [60 0 300:-60:120];