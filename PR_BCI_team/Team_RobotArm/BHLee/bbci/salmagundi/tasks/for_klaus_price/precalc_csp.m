%file= 'Guido_05_11_15/imag_lettGuido';
%file= 'Michael_05_11_10/imag_lettMichael';
%file= 'VPct_05_11_18/imag_lettVPct';
file= 'VPcm_06_02_07/imag_lettVPcm';

save_dir= [BCI_DIR 'tasks/for_klaus_price/'];

subdir= [fileparts(file) '/'];
is= min(find(subdir=='_'));
sbj= subdir(1:is-1);
sd= subdir;
sd(find(ismember(sd,'_/')))= [];

[cnt,mrk,mnt]= eegfile_loadMatlab(file);
if exist([EEG_MAT_DIR subdir 'imag_1drfb' sbj '.mat'], 'file'),
  bbci= eegfile_loadMatlab([subdir 'imag_1drfb' sbj], 'vars','bbci');
  classes= bbci.classes;
  csp_ival= bbci.setup_opts.ival;
  csp_clab= bbci.setup_opts.clab;
  csp_band= bbci.setup_opts.band;
  filt_b= bbci.analyze.csp_b;
  filt_a= bbci.analyze.csp_a;
else
  classes= mrk.className(1:2);
  csp_ival= [750 3500];
  csp_clab= {'not','E*','Fp*','AF*','I*','T9,10','TP9,10','FT9,10'};
  csp_band= [7 30];
  [filt_b,filt_a]= butter(5, csp_band/cnt.fs*2);
end

cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
cnt= proc_selectChannels(cnt, csp_clab);
cnt= proc_filt(cnt, filt_b, filt_a);
mrk= mrk_selectClasses(mrk, classes);

[mrk, rClab]= reject_eventsAndChannels(cnt, mrk, csp_ival, 'do_bandpass',0);
cnt= proc_selectChannels(cnt, 'not', rClab);

fv= makeEpochs(cnt, mrk, csp_ival);
[fv_csp, csp_w, csp_eig, csp_a]= ...
    proc_csp3(fv, 'patterns',3, 'scaling','maxto1');

csp_mnt= mnt_adaptMontage(mnt, cnt);
csp_sel= [1 size(csp_w,2)/2+1];
H= plotCSPanalysis(fv, csp_mnt, csp_w, csp_a, csp_eig, ...
                   'mark_patterns',csp_sel);

csp_w= csp_w(:,csp_sel);
csp_a= csp_a(csp_sel,:);
csp_clab= cnt.clab;
save([save_dir 'csp_' sd], 'csp_w', 'csp_a', 'csp_band', ...
     'csp_mnt', 'csp_clab', 'mrk');
