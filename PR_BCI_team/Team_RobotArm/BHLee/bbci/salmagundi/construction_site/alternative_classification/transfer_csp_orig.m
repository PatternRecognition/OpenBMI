res_dir= [DATA_DIR 'results/alternative_transfer/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

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

  %% load original settings that have been used for feedback
  bbci= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','bbci');
  %% get the two classes that have been used for feedback
  classes= bbci.classes;
  mrk= mrk_selectClasses(mrk, classes);
  %% get the time interval and channel selection that has been used
  csp_ival= bbci.setup_opts.ival;
  csp_clab= bbci.setup_opts.clab;
  filt_b= bbci.analyze.csp_b;
  filt_a= bbci.analyze.csp_a;
  csp_w= bbci.analyze.csp_w;
  
  %% preprocess data
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_filt(cnt, filt_b, filt_a);
  cnt= proc_linearDerivation(cnt, csp_w, 'prependix','csp');
  fv= makeEpochs(cnt, mrk, csp_ival);
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
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_linearDerivation(cnt, csp_w, 'prependix','csp');
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, S.mrk, [0 ilen]);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  memo{vp}.out= applyClassifier(fv, 'LSR', C);
  memo{vp}.label= fv.y;
  perf(vp)= loss_rocArea(fv.y, memo{vp}.out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));
end
save([res_dir 'csp_orig'], 'perf', 'memo', 'subdir_list');
