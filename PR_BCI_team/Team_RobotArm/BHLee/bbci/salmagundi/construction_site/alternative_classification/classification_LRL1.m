clear all;
cd ~;
startup;
cd([BCI_DIR '/construction_site/alternative_classification']);

addpath('/home/neuro/ryotat/lrl1/');
addpath('/home/neuro/ryotat/mutils/');

method  = 'LRL1';
res_dir= [DATA_DIR 'results/alternative_classification/'];

fprintf('target file [%s%s]\n', res_dir, method);

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

opt.test_ival= [1250 3500];

opt.clab = {'not','E*','Fp*','AF*','FT9,10','T9,10','TP9,10','OI*','I*'};
opt.filtOrder = 5;
opt.band = [7 30];
opt.ival = [500 3500];

model = struct('classy', {{'LRL1'  '*lin'  'solver'  'lrl1_dual'}},...
               'param', exp(linspace(log(0.01), log(100), 20)),...
               'xTrials', [2 10]);

memo=cell(1,length(subdir_list));


clear perf
for vp= 1:length(subdir_list),
  fprintf('analyzing [%s]...\n',subdir_list{vp});
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
  cnt= proc_selectChannels(cnt, opt.clab);
  [filt_b,filt_a]= butter(opt.filtOrder, opt.band/cnt.fs*2);
  cnt_lap= proc_laplace(cnt);
  csp_band= select_bandnarrow(cnt_lap, mrk_train, opt.ival);
 
  [filt_b,filt_a]= butter(opt.filtOrder, csp_band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);
  fv= makeEpochs(cnt, mrk_train, opt.ival);
  fv= proc_covariance(fv);
  
  [classy, loss, loss_std] = select_model(fv, model, 'output_all', 1, 'loss', 'rocArea');

  cls = trainClassifier(fv, classy);

  memo{vp}=archive('csp_band','classy','cls','loss','loss_std');
  
  %% evaluation of transfer on the second half of the calibration data
  %% load and preprocess data 
  cnt= proc_selectIval(cnt_memo, ival_test*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, opt.clab);
  cnt= proc_filt(cnt, filt_b, filt_a);
  fv= makeEpochs(cnt, mrk_test, opt.test_ival);
  fv= proc_covariance(fv);

  out= applyClassifier(fv, 'LRL1', cls);
  perf(vp)= loss_rocArea(fv.y, out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));

  memo{vp}.out = out;
end
save([res_dir method], 'perf', 'subdir_list', 'opt', 'memo');
