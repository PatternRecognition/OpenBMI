res_dir= [DATA_DIR 'results/alternative_classification/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

test_ival= [1250 3500];


clear perf
memo = cell(1,length(subdir_list));
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
  %% TODO: select channels
  csp_clab= bbci.setup_opts.clab;
  %{'not','Fp*','AF*','FT9,10','T9,10','TP9,10','OI*','I*'};
  cnt= proc_selectChannels(cnt, csp_clab);
  %[filt_b,filt_a]= butter(5, [7 30]/cnt.fs*2);
  %cnt_lap= proc_laplace(cnt);
  
  csp_ival= select_timeival(cnt, mrk_train);
  epo = makeEpochs(cnt,mrk_train,csp_ival);
  epo = proc_linearDerivation(epo,bbci.analyze.csp_w);
  fv = proc_fourierBandEnergy(epo,bbci.setup_opts.band);
  %[fv, sel_dims]= proc_fs_statistical(epo, 8, ...
%				      'method','fisherScore', ...
%				      'policy','number_of_features');
  C = trainClassifier(fv,'LSR');
  
  %% evaluation of transfer on the second half of the calibration data
  %% load and preprocess data 
  cnt= proc_selectIval(cnt_memo, ival_test*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_selectChannels(cnt, csp_clab);
  %cnt_lap= proc_laplace(cnt);
  %cnt_lap = proc_selectChannels(cnt_lap,'C3 lap','Cz lap','C4 lap');
  fv= makeEpochs(cnt, mrk_test, test_ival);
  fv = proc_linearDerivation(fv,bbci.analyze.csp_w);
  fv = proc_fourierBandEnergy(fv,bbci.setup_opts.band);
%  fv = proc_selectFeatures(fv,sel_dims);
  
  out= applyClassifier(fv, 'LSR', C);
  perf(vp)= loss_rocArea(fv.y, out);
  memo{vp}.out = out;
  memo{vp}.label = fv.y;
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));

end
save([res_dir 'fourier'], 'perf', 'subdir_list','memo');

