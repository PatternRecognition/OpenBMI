res_dir= [DATA_DIR 'results/alternative_classification/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

test_ival= [1250 3500];

model_RDA= strukt('classy', 'RDA', 'msDepth',2, 'inflvar',2);
model_RDA.param(1)= strukt('index',2, 'value',[0 0.05 0.25 0.5 0.75 0.9 1]);
model_RDA.param(2)= strukt('index',3, 'value',[0 0.001 0.01 0.1 0.3 0.5 0.7]);
%% RQDA:
%model_RDA= strukt('classy', {'RDA',0}, 'msDepth',2, 'inflvar',2);
%model_RDA.param= strukt('index',3, 'value',[0 0.001 0.01 0.1 0.3 0.5 0.7]);

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
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_filt(cnt, filt_b, filt_a);
  cnt= proc_linearDerivation(cnt, csp_w, 'prependix','csp');
  fv= makeEpochs(cnt, mrk_train, csp_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  classy= select_model(fv, model_RDA);
  C= trainClassifier(fv, classy);

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
  memo{vp}.out= applyClassifier(fv, classy, C);
  memo{vp}.label= fv.y;
  perf(vp)= loss_rocArea(fv.y, memo{vp}.out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));

end
save([res_dir 'csprda_orig'], 'perf', 'memo', 'subdir_list');
