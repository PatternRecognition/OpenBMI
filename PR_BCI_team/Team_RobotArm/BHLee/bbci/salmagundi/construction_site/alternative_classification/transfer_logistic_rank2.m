clear all;
cd ~;
startup;
cd([BCI_DIR '/construction_site/alternative_classification']);

addpath('/home/neuro/ryotat/csp/');
addpath('/home/neuro/ryotat/mutils/');
addpath('/home/neuro/ryotat/csp/optim/');

method  = 'logistic_rank2';
res_dir= [DATA_DIR 'results/alternative_transfer/'];

fprintf('target file [%s%s]\n', res_dir, method);

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

opt.clab = {'not','E*','Fp*','AF*','FT9,10','T9,10','TP9,10','OI*','I*'};
opt.filtOrder = 5;
opt.band = [7 30];
opt.ival = [500 3500];
opt.C     = exp(log(1.5)*(-19:0));
opt.msTrials = [2 5];

memo=cell(1,length(subdir_list));

clear perf stat
for vp= 1:length(subdir_list),
  fprintf('analyzing [%s]...\n',subdir_list{vp});
  
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
  
  %% preprocess data
  cnt= proc_selectChannels(cnt, opt.clab);
  [filt_b,filt_a]= butter(opt.filtOrder, opt.band/cnt.fs*2);
  cnt_lap= proc_laplace(cnt);
  csp_band= select_bandnarrow(cnt_lap, mrk, opt.ival);
  clear cnt_lap

  [filt_b,filt_a]= butter(opt.filtOrder, csp_band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);
  fv= makeEpochs(cnt, mrk, opt.ival);

  [loss, loss_std] = xvalLogistic_rank2(fv, opt.msTrials, opt.C);
  [m,ixModel]=min(loss);
  [epocv, Ww] = proc_whiten(proc_covariance(fv));
  [fv, W, bias]  = logistic_rank2(epocv, [], opt.C(ixModel));

  W = Ww*W;
  C=struct('w',.5*[-1;1],'b',bias);

  memo{vp}=archive('csp_band','W','bias','ixModel','loss','loss_std');

  %% evaluation of transfer: calibration -> feedback
  %% load and preprocess data of feedback session
  cnt= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','cnt');
  S= load([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '_mrk_1000']);
  mrk= S.mrk;  %% use markers for short-term windows
  ilen= S.opt_stw.win_len;
  cnt= proc_selectChannels(cnt, opt.clab);
  cnt= proc_linearDerivation(cnt, W, 'prependix','csp');
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, mrk, [0 ilen]);
  fv= proc_variance(fv);
  
  out= applyClassifier(fv, 'LSR', C);
  perf(vp)= loss_rocArea(fv.y, out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));
  
  idx1= find(fv.y(1,:));
  idx2= find(fv.y(2,:));
  stat(vp, 1:8)= [mean(out(idx1)), mean(out(idx2)), ...
                  std(out(idx1)), std(out(idx2)), ...
                  skewness(out(idx1)), skewness(out(idx2)), ...
                  kurtosis(out(idx1)), kurtosis(out(idx2))];

  memo{vp}.out=out;
end
save([res_dir method], 'perf', 'stat', 'subdir_list', 'opt','memo');
