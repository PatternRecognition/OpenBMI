dir_list= {'Gabriel_04_03_30', ...
           'Guido_04_03_29', ...
           'Falk_04_03_31', ...
           'Matthias_04_03_24', ...
           'Klaus_04_04_08'};
code_list= {'aa','al','av','aw','ay'};
train_perc= [60 80 30 20 10];

classes= {'right', 'foot'};
klcsp.clab = {'not','E*'};
csp.ival= [500 3750];
csp.band= [7 30];

for dd= 1:length(dir_list),
  dir_name= dir_list{dd};
  is= find(dir_name=='_');
  sbj= dir_name(1:is-1);
  file= strcat(dir_name, '/', {'imag_move', 'imag_lett'}, sbj);
  [cnt, mrk]= eegfile_loadMatlab(file, 'clab',csp.clab);
  mrk= mrk_selectClasses(mrk, classes);
  nEpochs= length(mrk.y);
  tp= ceil(train_perc(dd)/100*nEpochs);
  mrk_train= mrk_chooseEvents(mrk, 1:tp);
  mrk_test= mrk_chooseEvents(mrk, tp+1:nEpochs);
%  [size(mrk_train.y,2) size(mrk_test.y,2)]
  
  [b, a]= butter(5, csp.band/cnt.fs*2);
  cnt_flt= proc_filt(cnt, b, a);
  
  fv= cntToEpo(cnt_flt, mrk_train, csp.ival);
%  [fv,csp_w]= proc_csp_auto(fv, 3);
  [fv,csp_w]= proc_csp3(fv, 3);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  C= trainClassifier(fv, 'LDA');
  
  fv= cntToEpo(cnt_flt, mrk_test, csp.ival);
  fv= proc_linearDerivation(fv, csp_w);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  out= applyClassifier(fv, 'LDA', C);
  fprintf('%s: %.1f%%\n', code_list{dd}, 100*mean(sign(out)==[-1 1]*fv.y));
end
