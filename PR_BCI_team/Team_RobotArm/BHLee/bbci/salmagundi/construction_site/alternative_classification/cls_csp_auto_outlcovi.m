res_dir= [DATA_DIR 'results/alternative_classification/temp/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

test_ival= [1250 3500];


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
  
  %% determine split of the data into training and test set
  %% chronological split: 1st half for training, 2nd half for testing
  nEvents= length(mrk.pos);
  idx_train= 1:ceil(nEvents/2);
  idx_test= ceil(nEvents/2)+1:nEvents;
  ival_train= [0 mrk.pos(idx_train(end))+5*mrk.fs];
  ival_test= [mrk.pos(idx_test([1 end])) + [-1 5]*mrk.fs];
  mrk_train= mrk_selectEvents(mrk, idx_train);
  mrk_test= mrk_selectEvents(mrk, idx_test);
  mrk_test.pos= mrk_test.pos - ival_test(1);
  cnt_memo= cnt;
  
  %% process training data and train classifier
  cnt= proc_selectIval(cnt_memo, ival_train*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  csp_clab= {'not','Fp*','AF*','FT9,10','T9,10','TP9,10','OI*','I*'};
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt_lap= proc_laplace(cnt);
  [filt_b,filt_a]= butter(5, [7 30]/cnt.fs*2);
  cnt_flt= proc_filt(cnt_lap, filt_b, filt_a);
  csp_ival= select_timeival(cnt_flt, mrk_train);
  clear cnt_flt
  csp_band= select_bandnarrow(cnt_lap, mrk_train, csp_ival);
  clear cnt_lap
  [filt_b,filt_a]= butter(5, csp_band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);
  csp_ival= select_timeival(cnt, mrk_train);
  fv= makeEpochs(cnt, mrk_train, csp_ival);
  [fv, outl_idx, dist]= ...
      proc_outl_cov(fv, 'display',1, 'iterative',1);
  drawnow;
  fprintf('%d outliers rejected\n', length(outl_idx));
  
  [fv, csp_w]= proc_csp3(fv, 'patterns',3, 'scaling','maxto1');
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  %% TODO: select CSP filters
  C= trainClassifier(fv, 'LSR');

  %% evaluation of transfer on the second half of the calibration data
  %% load and preprocess data 
  cnt= proc_selectIval(cnt_memo, ival_test*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_linearDerivation(cnt, csp_w, 'prependix','csp');
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, mrk_test, test_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  memo{vp}.out= applyClassifier(fv, 'LSR', C);
  memo{vp}.label= fv.y;
  perf(vp)= loss_rocArea(fv.y, memo{vp}.out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));

end
save([res_dir 'csp_auto_outlcovi'], 'perf', 'memo', 'subdir_list');
