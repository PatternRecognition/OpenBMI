%%
% run('D:\svn\bbci\toolbox\startup\startup_tubbci2.m')
%  VP_CODE = 'VPjg';
% global TODAY_DIR REMOTE_RAW_DIR
% acq_getDataFolder;%('log_dir',1);
% REMOTE_RAW_DIR= TODAY_DIR;
%%
Para.fs =44100;
Para.act_time = 2;
Para.ref_time = 0.5;
Para.modfreq = [21];
Para.carfreq = 200;
Para.num_trial = 20;
Para.count_dura = 8;
Para.ifi = 0.5;
Para.num_block = 4;
Para.filename='test_fingertipright_21_200_lev6_ampmod';
save(['D:\svn\bbci\acquisition\stimulation\master_fouad\Para_',Para.filename,VP_CODE],'Para')

opt= [];
opt.position= [-1919 0 1920 1200];
opt.msg_intro = strvcat('Please try to sit still ','and focus on the screen.');
opt.countdown_msg = 'Be prepared: Start in %d s';
opt.filename= 'imag_arrow';
opt.breaks= [15 15];  %% Alle 15 Stimuli Pause fuer 15 Sekunden
opt.break_msg= 'Short break (%d s)';

opt.cross_vpos=0.10;
opt.cross_spec = {'Color',[0 0 0], 'LineWidth',4};
opt.cross_size =0.13;
% 3+(2+0.5)*length(Para.modfreq)
StimTuningfunc_ver2(Para,opt)
% [zero_mat_ref,sig_pau_ref,sig_pau,sig,sig_trial, mod_freq_out] = plotStimTunfunc(Para);
