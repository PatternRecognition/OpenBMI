% please be careful this script takes forever to
% finish....
% I calculated the spectra independently..
%
% sf @ida

clear all
startup_bci

res_dir= [DATA_DIR 'results/alternative_classification/'];

subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

test_ival= [1250 3500];

clear perf
for vp= 1:length(subdir_list),
  
  clear spec_arma_tr spec_arma_te spec_ts_tr spec_ts_te spec phase_ts_tr ...
    phase_ts_te spec_ts
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
  %cnt_memo.x=diff(cnt_memo.x);
  %% process training data and train classifier
  cnt= proc_selectIval(cnt_memo, ival_train*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  cnt= proc_selectChannels(cnt, csp_clab);
  %cnt= proc_filt(cnt, filt_b, filt_a);
  cnt= proc_linearDerivation(cnt, csp_w, 'prependix','csp');
  fv= makeEpochs(cnt, mrk_train, csp_ival);
  fv_tr=fv; 
  
  
  freq_range=bbci.setup_opts.band(1):1:bbci.setup_opts.band(2);
  fs=100;
  interval=100;
  running=size(fv.x,1)-interval;
  [Nti Nch Ntr] =size(fv.x);
  
  for trNo=1:1:Ntr
    for chNo=1:1:Nch
      % calc for all of signal
      [ar,ma]=sig2arma(fv.x(:,chNo,trNo));
      dum=freqz(ma,ar,freq_range,fs);
      spec_arma_tr(:,chNo,trNo)=10*log10(abs(dum)+eps);
      model_order=length(ma);
      clear dum ar ma
      for i=1:1:running
	data=fv.x(i:(interval+i),chNo,trNo);
	[ar,ma]=sig2arma(data,model_order);
	spec=freqz(ma,ar,freq_range,fs);
	phase_ts_tr(i,:,chNo,trNo)=atan(imag(spec)./real(spec));
	spec_ts(i,:,chNo,trNo)=10*log10(abs(spec)+eps);
	clear ar ma data
      end
    end
    trNo
  end  
 
  spec_ts_tr=fv;
  spec_ts_tr.x=spec_ts;
 
  %% evaluation of transfer on the second half of the calibration data
  %% load and preprocess data 
  cnt= proc_selectIval(cnt_memo, ival_test*1000/mrk.fs);
  cnt= proc_selectChannels(cnt, 'not','E*');
  cnt= proc_selectChannels(cnt, csp_clab);
  cnt= proc_linearDerivation(cnt, csp_w, 'prependix','csp');
  %cnt= proc_filt(cnt, filt_b, filt_a);
  
  fv= makeEpochs(cnt, mrk_test, csp_ival);
  fv_te=fv;
  
  
  interval=100;
  running=size(fv.x,1)-interval;
  [Nti Nch Ntr] =size(fv.x);
  clear dum
  for trNo=1:1:Ntr
    for chNo=1:1:Nch
      % calc for all of signal
      [ar,ma]=sig2arma(fv.x(:,chNo,trNo));
      dum=freqz(ma,ar,freq_range,fs);
      spec_arma_te(:,chNo,trNo)=10*log10(abs(dum)+eps);
      model_order=length(ma);
      clear dum ar ma
      for i=1:1:running
	data=fv.x(i:(interval+i),chNo,trNo);
	[ar,ma]=sig2arma(data,model_order);
	spec=freqz(ar,ma,freq_range,fs);
	phase_ts_te(i,:,chNo,trNo)=atan(imag(spec)./real(spec));
	spec_ts(i,:,chNo,trNo)=10*log10(abs(spec)+eps);
	clear ar ma data
      end
    end
    trNo
  end  
   
  spec_ts_te=fv;
  spec_ts_te.x=spec_ts;
 
  save(['~/mycode/alt_class/data/arma_ts',num2str(vp),'.mat'],'spec_ts_tr','spec_ts_te','fv_tr','fv_te','spec_arma_tr','spec_arma_te','phase_ts_tr','phase_ts_te') 
end



clear perf1 perf2
for i=1:1:25
  if i~=5
    
    %i=1; 
    load(['~/mycode/alt_class/data/arma_ts',num2str(i),'.mat'])
    
    % armasel
    dum1=spec_arma_tr;
    dum2=spec_arma_te;
    spec_arma_tr=fv_tr;
    spec_arma_te=fv_te;
    spec_arma_tr.x=dum1;
    spec_arma_te.x=dum2;
    clear dum1 dum2
    
    spec_arma_tr=proc_normalize(spec_arma_tr);
    
    % time-series
    [ti,fr,ch,tr]=size(spec_ts_tr.x);
    
    spec_ts_tr.x=reshape(spec_ts_tr.x,fr*ti,ch,tr);
    spec_ts_te.x=reshape(spec_ts_te.x,fr*ti,ch,tr);
    spec_ts_tr=proc_normalize(spec_ts_tr);
    %spec_ts_tr=proc_baseline(spec_ts_tr);
    
    C1= trainClassifier(spec_ts_tr, 'LSR');
    C2= trainClassifier(spec_arma_tr, 'LSR');
    
    
    
    
    out1= applyClassifier(spec_ts_te, 'LSR', C1);
    out2=applyClassifier(spec_arma_te, 'LSR', C2);
    perf1(i)= loss_rocArea(spec_ts_te.y, out1);
    perf2(i)= loss_rocArea(spec_arma_te.y, out2);
    fprintf('%4.1f%%', 100-(100*perf1(i)));
    fprintf('%4.1f%%\n', (100*perf2(i)));
  end
end
subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');
res_dir= [DATA_DIR 'results/alternative_classification/'];

perf1=perf1(find(perf1~=0));
perf1=1-perf1;
mean_perf1=mean(perf1)*100
perf=perf1;
save([res_dir 'armaSpecCspTS'], 'perf', 'subdir_list');
clear perf
perf2=perf2(find(perf2~=0));
%perf2=1-perf2;
perf=perf2;
mean_perf2=mean(perf2)*100
save([res_dir 'armaSpecCsp'], 'perf', 'subdir_list');

