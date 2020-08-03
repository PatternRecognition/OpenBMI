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
  [mrk_clean, rClab, dmy, nfo]= ...
      reject_varEventsAndChannels(cnt, mrk_train, [500 4500]);
  fprintf('rejected: /%s/ -> %d trials and channels /', ...
          vec2str(apply_cellwise2(nfo.trials, 'length'), '%d', '/'), ...
          length(mrk_train.pos)-length(mrk_clean.pos));
  for cc= 1:length(nfo.chans),
    fprintf('%s/', vec2str(cnt.clab(nfo.chans{cc}),'%s',','));
  end
  fprintf('\n');
  cnt= proc_selectChannels(cnt, 'not', rClab);
  cnt_lap= proc_laplace(cnt);
  [filt_b,filt_a]= butter(5, [7 30]/cnt.fs*2);
  cnt_flt= proc_filt(cnt_lap, filt_b, filt_a);
  csp_ival= select_timeival(cnt_flt, mrk_clean);
  clear cnt_flt
  csp_band= select_bandnarrow(cnt_lap, mrk_clean, csp_ival);
  clear cnt_lap
  [filt_b,filt_a]= butter(5, csp_band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);
  csp_ival= select_timeival(cnt, mrk_clean);
  epo= makeEpochs(cnt, mrk_clean, csp_ival);
  nTrials= size(epo.x,3);
  all_outliers_removed= 0;
  while ~all_outliers_removed & size(epo.x,3)>0.85*nTrials,
    [fv, csp_w]= proc_csp3(epo, 'patterns',3, 'scaling','maxto1');
    %% find loo csp-outliers
    wckflt= zeros([6 size(fv.x,3)]);
    for tr= 1:size(fv.x,3),
      epo_loo= proc_selectEpochs(epo, 'not', tr);
      [dmy, csp_w_loo, dmy, csp_a_loo]= ...
          proc_csp3(epo_loo, 'scaling','maxto1', ...
                    'selectPolicy','matchfilters', 'patterns',csp_w);
      A= csp_w'*csp_w_loo./sqrt(csp_w'*csp_w)./sqrt(csp_w_loo'*csp_w_loo);
      wckflt(:,tr)= 100*(1-abs(diag(A)));
    end
    %% criterium: larger than five times the 80%ile value (or >10)
    kickout= cell(1, size(wckflt,1));
    perc80= percentiles(wckflt, 80);
    thresh= max(2, min(10, 5*perc80));
    for cc= 1:size(wckflt,1),
      kickout{cc}= find( wckflt(cc,:) > thresh(cc) );
    end
    fprintf('[%s]\n', vec2str(thresh, '%5.2f', '  '));
    ik= apply_cellwise2(kickout, 'length');
    iBadFilter= find(ik>5);
    if ~isempty(iBadFilter),
      ik(iBadFilter)= NaN;
      kickout(iBadFilter)= {[]};
    end
    fprintf('[%s] -> %d\n', vec2str(ik, '%5d', '  '), length([kickout{:}]));
    if length([kickout{:}])>0,
      epo= proc_selectEpochs(epo, 'not', [kickout{:}]);
    else
%      if size(epo.x,3)>0.9*nTrials,
%        thresh= max(0.25, 2*perc80);
%        for cc= 1:size(wckflt,1),
%          kickout{cc}= find( wckflt(cc,:) > thresh(cc) );
%        end
%        epo= proc_selectEpochs(epo, 'not', [kickout{:}]);
%      end
      all_outliers_removed= 1;
    end
  end
  %% TODO: discard BadFilters (see above)
  [fv, csp_w]= proc_csp3(epo, 'patterns',3, 'scaling','maxto1');
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  C= trainClassifier(fv, 'LSR');

  %% evaluation of transfer on the second half of the calibration data
  %% load and preprocess data 
  cnt= proc_selectIval(cnt_memo, ival_test*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_selectChannels(cnt, 'not', rClab);
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
save([res_dir 'csp_auto_outl'], 'perf', 'memo', 'subdir_list');
