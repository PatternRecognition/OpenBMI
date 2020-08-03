N= 150;

opt= struct('perc_dev', 40/100);
opt.avoid_dev_repetitions= 1;
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 2000;
opt.isi_jitter= 250;

opt.cue_dev= strcat('bfnt_a2_season2_7222_', cprintf('%01d_', selected_set),cprintf('%01d_', word_no),cprintf('%01d', speaker_no));
opt.cue_std= strcat('bfnt_a2_season2_std_',cprintf('%01d_', word_no),cprintf('%01d', speaker_no));
