date_list= {'04_03_24', ...
            '04_03_29', ...
            '04_03_30', ...
            '04_03_31', ...
            '04_04_08'};

feedback_files = {'imag_lett1Matthias_udp_1d_szenario_02',...
                  'imag_lettGuido_udp_1d_szenario_01',...
                  'imag_moveGabriel_udp_1d_szenario_02',...
                  'imag_lettFalk_udp_1d_szenario_01',...
                  'imag_lettKlaus_udp_1d_szenario_01',...
                  };

expbase= expbase_read;
imagtrain= expbase_select(expbase, 'paradigm',{'imag_lett', 'imag_move'});
imagtrain= expbase_joinParadigms(imagtrain, 'imag_lett', 'imag_move');
imagcurs= expbase_select(expbase, 'paradigm', 'imag_curs');

for ff= 1:length(date_list),
  %% load EEG data
  train_file= expbase_select(imagtrain, 'date',date_list{ff});
  file= expbase_filename(train_file);
  [cnt, mrk, mnt]= eegfile_loadMatlab(strcat(file, '_16bit'));
  
  %% load set proprocessing settings
  szenario= feedback_files{ff};
  S= load([EEG_RAW_DIR train_file.subject '_' train_file.date '/' szenario]);
  classes= S.classes;
  csp= S.csp;
  csp.filt_b= S.dscr.proc_cnt_apply.param{1};
  csp.filt_a= S.dscr.proc_cnt_apply.param{2};
  
  cnt= proc_selectChannels(cnt, csp.clab);
  cnt= proc_filt(cnt, csp.filt_b, csp.filt_a);
  mrk_cl= mrk_selectClasses(mrk, classes);
  fv= makeEpochs(cnt, mrk_cl, csp.ival);
  
  proc= struct('memo', {{'csp_w'}});
  proc.train= ['[fv,csp_w]= proc_csp2(fv,' int2str(csp.nPat) '); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
  proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
               
  %% validate performance on the training data in three ways
  [loss1(ff), lstd, out]= ...
      xvalidation(fv, 'LDA', 'proc',proc, 'sample_fcn','leaveOneOut');
  iserr= sign(out)~=[-1 1]*fv.y;
  err= 100*mean(iserr)
  plot(iserr, '.');
  hold on;
  %% there is something strange with the function movingAverage
  err_ma= movingAverage(iserr', 15, 'centered');
  plot(err_ma, 'LineWidth',2);
  hold off;
  set(gca, 'YLim',[-0.1 1.1]);
  title(sprintf('errors of %s (leave-one-out)', train_file.subject));

  loss2(ff)= xvalidation(fv, 'LDA', 'proc',proc, 'sample_fcn','evenOdd');
  [loss3(ff), lstd, out]= ...
      xvalidation(fv, 'LDA', 'proc',proc, 'sample_fcn','chronSplit');
  iValid= find(~isnan(out));
  iserr= sign(out(iValid))~=[-1 1]*fv.y(:,iValid);
  err= 100*mean(iserr)
  plot(iserr, '.');
  hold on;
  err_ma= movingAverage(iserr', 15, 'centered');
  plot(err_ma, 'LineWidth',2);
  hold off;
  set(gca, 'YLim',[-0.1 1.1]);
  title(sprintf('errors of %s (chron-split)', train_file.subject));

  fprintf('%s: loo: %.1f%%, even-odd: %.1f%%, chron-split: %.1f%%\n', ...
          train_file.subject, 100*loss1(ff), 100*loss2(ff), 100*loss3(ff));

  %% validate classifier from taining data on other data
  eval(proc.train);
  C= trainClassifier(fv, 'LDA');
  
  curs_file= expbase_select(imagcurs, 'date',date_list{ff});
  file= expbase_filename(curs_file);
  [ct,mk]= eegfile_loadMatlab(strcat(file, '_16bit'));
  ct= proc_selectChannels(ct, csp.clab);
  ct= proc_filt(ct, csp.filt_b, csp.filt_a);
  mk= mrk_selectClasses(mk, classes);
  fv= makeEpochs(ct, mk, csp.ival);
  eval(proc.apply);
  
  out= applyClassifier(fv, 'LDA', C);
  iserr= sign(out)~=[-1 1]*fv.y;
  err= 100*mean(iserr)
  plot(iserr, '.');
  hold on;
  err_ma= movingAverage(iserr', 15, 'centered');
  plot(err_ma, 'LineWidth',2);
  hold off;
  set(gca, 'YLim',[-0.1 1.1]);
  title(sprintf('errors in %s', untex(file)));
  
end
