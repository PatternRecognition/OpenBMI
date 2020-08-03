addpath([IMPORT_DIR 'eeglab5.02/functions/']);
res_dir= [DATA_DIR 'results/alternative_transfer/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');


selection_nComp= 8;
selection_policy= 'roc';
time_ival= [1250 3500];


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
  %% get the channel selection that has been used
  csp_clab= bbci.setup_opts.clab;
  
  %% process training data and train classifier
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt_lap= proc_laplace(cnt);
  band= select_bandnarrow(cnt_lap, mrk, time_ival);
  [filt_b,filt_a]= butter(5, band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);

  [ww,sph] = runica(cnt.x', 'maxsteps',10, 'verbose','off');
  ica_w_full= (ww*sph)';

  cnt= proc_linearDerivation(cnt, ica_w_full, 'prependix','ica');
  fv= makeEpochs(cnt, mrk, time_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);

  %% select good ICA components
  switch(lower(selection_policy)),
   case 'fisherscore',
    [fvs, idx]= proc_fs_statistical(fv, selection_nComp, ...
                                    'method','fisherScore', ...
                                    'policy','number_of_features');
   case 'roc',
    [fvs, idx]= proc_fs_statistical(fv, selection_nComp, ...
                                    'method','rocAreaValues', ...
                                    'policy','number_of_features');
   case 'lpm',
    model_LPM= struct('classy','LPM', 'msDepth',2, 'std_factor',2);
    model_LPM.param= struct('index',2, 'scale','log', ...
                            'value', [-2:6]);
    [fvs, idx]= proc_fs_classifierWeights(fv, selection_nComp, model_LPM, ...
                                          'policy','number_of_features');
   case 'xval',
    [fvs, idx]= proc_fs_byXvalidation(fv, selection_nComp, ...
                                      'model','LDA', ...
                                      'policy','number_of_features');
   case 'xval_inc',
    [fvs, idx]= proc_fs_byXvalidation(fv, selection_nComp, ...
                                      'model','LDA', ...
                                      'method','incremental', ...
                                      'policy','number_of_features');
   case 'xval_dec',
    model_RLDA= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
    model_RLDA.param= [0 0.001 0.005 0.01 0.05 0.1 0.5];
    [fvs, idx]= proc_fs_byXvalidation(fv, selection_nComp, ...
                                      'model',model_RLDA, ...
                                      'method','decremental', ...
                                      'policy','number_of_features');
   otherwise,
    error('selection policy not known');
  end
  selected(vp)= {idx};
  ica_w= ica_w_full(:,idx);
  fv= proc_selectChannels(fv, idx);
  C= trainClassifier(fvs, 'LSR');

  %% evaluation of transfer: calibration -> feedback
  %% load and preprocess data of feedback session
  cnt= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','cnt');
  S= load([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '_mrk_1000']);
  mrk= S.mrk;  %% use markers for short-term windows
  ilen= S.opt_stw.win_len;
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_linearDerivation(cnt, ica_w, 'prependix','ica');
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, S.mrk, [0 ilen]);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  memo{vp}.out= applyClassifier(fv, 'LSR', C);
  memo{vp}.label= fv.y;
  perf(vp)= loss_rocArea(fv.y, memo{vp}.out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));
end
save([res_dir 'ica_infomax_' selection_policy '_' int2str(selection_nComp)], ...
     'perf', 'memo', 'subdir_list');
