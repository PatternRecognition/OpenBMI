Para.fs = 22050;
Para.act_time = 2;
Para.ref_time = 2;
Para.modfreq = [5:2:31];
Para.carfreq = 200;
Para.num_trial = 10;
Para.count_dura = 10;
Para.ifi = 1;
Para.num_block = 4;

StimTuningfunc_ver2(Para)
% [zero_mat_ref,sig_pau_ref,sig_pau,sig,sig_trial, mod_freq_out] = plotStimTunfunc(Para);
