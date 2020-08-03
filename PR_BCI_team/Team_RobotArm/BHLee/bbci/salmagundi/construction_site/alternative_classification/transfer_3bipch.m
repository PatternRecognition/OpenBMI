res_dir= [DATA_DIR 'results/alternative_transfer/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

time_ival= [1250 3500];
clab= {'FC*', 'CFC*', 'C*', 'CPC*', 'CP*', 'PCP*', 'P*'};

clear perf stat
for vp= 1:length(subdir_list),

  sub_dir= [subdir_list{vp} '/'];
  is= min(find(sub_dir=='_'));
  sbj= sub_dir(1:is-1);

  if ~exist([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '.mat'], 'file'),
    perf(vp)= NaN;
    stat(vp,1:8)= NaN;
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
  %% get the filter that has been used
  filt_b= bbci.analyze.csp_b;
  filt_a= bbci.analyze.csp_a;
  
  %% preprocess data
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  cnt= proc_bipolarChannels(cnt, bipclab);
  cnt= proc_filt(cnt, filt_b, filt_a);
  fv= makeEpochs(cnt, mrk, time_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  C= trainClassifier(fv, 'LSR');

  %% evaluation of transfer: calibration -> feedback
  %% load and preprocess data of feedback session
  cnt= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','cnt');
  S= load([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '_mrk_1000']);
  mrk= S.mrk;  %% use markers for short-term windows
  ilen= S.opt_stw.win_len;
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_bipolarChannels(cnt, bipclab);
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, S.mrk, [0 ilen]);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  out= applyClassifier(fv, 'LSR', C);
  perf(vp)= loss_rocArea(fv.y, out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));
  
  idx1= find(fv.y(1,:));
  idx2= find(fv.y(2,:));
  stat(vp, 1:8)= [mean(out(idx1)), mean(out(idx2)), ...
                  std(out(idx1)), std(out(idx2)), ...
                  skewness(out(idx1)), skewness(out(idx2)), ...
                  kurtosis(out(idx1)), kurtosis(out(idx2))];

end
save([res_dir '3bipch'], 'perf', 'stat', 'subdir_list');
