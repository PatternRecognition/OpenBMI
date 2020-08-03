% niedrigeres N gewählt (80 statt 100)
N= 80;
calib_set= [660 1265 1585 1985;
            885 1425 1825 2305;
];

opt= struct('perc_dev', 40/100);
opt.avoid_dev_repetitions= 1;
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 2000;
opt.isi_jitter= 250;
opt.filename= sprintf('bfnt_a3_season2_calib_set%d%d%d', set_no, word_no, speaker_no);

opt.cue_dev= strcat('bfnt_a2_season2_7222_', cprintf('%01d_', calib_set(set_no,:)),cprintf('%01d_', word_no),cprintf('%01d', speaker_no));
opt.cue_std= strcat('bfnt_a2_season2_std_',cprintf('%01d_', word_no),cprintf('%01d', speaker_no));