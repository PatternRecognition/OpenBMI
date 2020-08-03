res_dir= [DATA_DIR 'results/alternative_transfer/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

time_ival= [1250 3500];
clab= {'FC*', 'CFC*', 'C*', 'CPC*', 'CP*', 'PCP*', 'P*'};

clear perf
memo=cell(1,length(subdir_list));

for vp= 1:length(subdir_list),

  sub_dir= [subdir_list{vp} '/'];
  is= min(find(sub_dir=='_'));
  sbj= sub_dir(1:is-1);

  if ~exist([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '.mat'], 'file'),
    perf(vp)= NaN;
    continue;
  end

  %% load data of calibration (training) session
  if strcmp(sbj, 'VPco'),
    [cnt,mrk,mnt]= eegfile_loadMatlab([sub_dir 'real' sbj]);
  else
    [cnt,mrk,mnt]= eegfile_loadMatlab([sub_dir 'imag_lett' sbj]);
  end
  %% get the two classes that have been used for feedback
  bbci= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','bbci');
  classes= bbci.classes;
  mrk= mrk_selectClasses(mrk, classes);
  
  %% preprocess data
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  cnt= proc_selectChannels(cnt, getClabForLaplace(cnt, clab));
  [cnt,lap_w]= proc_laplace(cnt);
  band= select_bandnarrow(cnt, mrk, time_ival);
  [filt_b,filt_a]= butter(5, band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);
  fv= makeEpochs(cnt, mrk, time_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  [fvs, sel_clab]= proc_fs_statistical(fv, 8, ...
                                       'method','fisherScore', ...
                                       'policy','number_of_features');
  fv= proc_selectChannels(fv, sel_clab);
  C= trainClassifier(fv, 'LSR');

  %% evaluation of transfer: calibration -> feedback
  %% load and preprocess data of feedback session
  cnt= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','cnt');
  S= load([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '_mrk_1000']);
  mrk= S.mrk;  %% use markers for short-term windows
  ilen= S.opt_stw.win_len;
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_selectChannels(cnt, getClabForLaplace(cnt, clab));
  cnt= proc_linearDerivation(cnt, lap_w, 'prependix','lap');
  cnt= proc_selectChannels(cnt, sel_clab);
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, S.mrk, [0 ilen]);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  memo{vp}.out= applyClassifier(fv, 'LSR', C);
  memo{vp}.label= fv.y;
  perf(vp)= loss_rocArea(fv.y, memo{vp}.out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));
end
save([res_dir 'laplace'], 'perf', 'memo', 'subdir_list');
