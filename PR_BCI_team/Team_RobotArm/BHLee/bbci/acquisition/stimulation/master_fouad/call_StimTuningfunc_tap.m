%%
% run('D:\svn\bbci\toolbox\startup\startup_tubbci2.m')
%  VP_CODE = 'VPjw';
% global TODAY_DIR REMOTE_RAW_DIR
% acq_getDataFolder('log_dir',1);
% REMOTE_RAW_DIR= TODAY_DIR;
%%
Para.fs = 44100;
Para.act_time = 2;
Para.ref_time =5;
Para.modfreq = [17:2:33];
Para.carfreq = 200;
Para.num_trial = 10;
Para.count_dura = 7;
Para.ifi = 0.5;
Para.num_block = 8;
Para.duty = 1/5;
Para.off =4/5;
Para.filename='fingertipRight';
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
StimTuningfunc_tap(Para,opt)

% [zero_mat_ref,sig_pau_ref,sig_pau,sig,sig_trial, mod_freq_out] = plotStimTunfunc(Para);
