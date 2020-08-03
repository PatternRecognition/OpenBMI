opt.cuename ='cue_';
opt.fs         = 44100;
opt.act_time   = 5;
opt.carr_freq    = 200;
opt.final_time =4.2;
opt.twitch_start=3.7;
opt.twitch_dura =[];
opt.twitch_amp =0.2;
 for i=17:2:33,
 opt.mod_freq = i;
[sig t_long t_final] = createTapStimuli(opt,SOUND_DIR);
 end


