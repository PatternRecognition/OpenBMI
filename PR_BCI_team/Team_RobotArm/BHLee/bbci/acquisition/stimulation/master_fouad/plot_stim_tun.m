function plot_stim_tun(sig_trial,fs,active_time,ref_time,num_freq)

num_trial = size(sig_trial,1);
length_trial = ((active_time*num_freq)+ref_time);
t_trial = [0:1/fs:(length_trial)-1/fs];



