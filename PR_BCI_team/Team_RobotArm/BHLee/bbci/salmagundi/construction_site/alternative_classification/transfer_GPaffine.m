clear all;
cd ~;
startup;
cd([BCI_DIR '/construction_site/alternative_classification']);

addpath('/home/neuro/ryotat/csp/');
addpath('/home/neuro/ryotat/mutils/');

method  = 'GPaffine';
res_dir= [DATA_DIR 'results/alternative_transfer/'];

fprintf('target file [%s%s]\n', res_dir, method);

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

opt.clab = {'not','E*','Fp*','AF*','FT9,10','T9,10','TP9,10','OI*','I*'};
opt.filtOrder = 5;
opt.band = [7 30];
opt.ival = [500 3500];

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
  epo= makeEpochs(cnt, mrk, opt.ival);

  [epocv, Ww] = proc_whiten(proc_covariance(epo));

  cl=train_classifierGP(squeeze(epocv.x), epocv.y,...
                       'kernel', {'affine','alpha', 0}, ...
                        'scaledata',0,...
    'verbosity',2,'optimizer', {'opt_BFGS', 'lower', [-700 nan], 'upper', [nan 700]});
  
  % {'opt_BFGS', 'checkGradient', 1}
  
  alpha = cl.opt.kernel{1}{2}.alpha;
  kernelweight = cl.opt.kernelweight;
  memo{vp}=archive('csp_band','Ww','alpha','kernelweight');

  %% evaluation of transfer: calibration -> feedback
  %% load and preprocess data of feedback session
  cnt= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','cnt');
  S= load([EEG_MAT_DIR sub_dir 'imag_1drfb' sbj '_mrk_1000']);
  mrk= S.mrk;  %% use markers for short-term windows
  ilen= S.opt_stw.win_len;
  cnt= proc_selectChannels(cnt, opt.clab);
  cnt= proc_linearDerivation(cnt, Ww, 'prependix','csp');
  cnt= proc_filt(cnt, filt_b, filt_a);

  fv= makeEpochs(cnt, mrk, [0 ilen]);
  fv= proc_covariance(fv);
  
  [p out out_v]= apply_classifierGP(cl, squeeze(fv.x));
  perf(vp)= loss_rocArea(fv.y, out);
  fprintf('%10s  ->  %4.1f%%\n', sbj, 100*perf(vp));
  
  idx1= find(fv.y(1,:));
  idx2= find(fv.y(2,:));
  stat(vp, 1:8)= [mean(out(idx1)), mean(out(idx2)), ...
                  std(out(idx1)), std(out(idx2)), ...
                  skewness(out(idx1)), skewness(out(idx2)), ...
                  kurtosis(out(idx1)), kurtosis(out(idx2))];

  memo{vp}.out=out;
  memo{vp}.out=out_v;
end
save([res_dir method], 'perf', 'stat', 'subdir_list', 'opt','memo');
