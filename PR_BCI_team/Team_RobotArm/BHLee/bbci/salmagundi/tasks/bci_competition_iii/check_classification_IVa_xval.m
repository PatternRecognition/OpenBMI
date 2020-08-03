dir_list= {'Falk_04_03_31', ...
           'Gabriel_04_03_30', ...
           'Guido_04_03_29', ...
           'Klaus_04_04_08', ...
           'Matthias_04_03_24'};

classes= {'right', 'foot'};
csp.clab = {'not','E*','Fp*','FAF*','I*','AF*'};
csp.ival= [750 3750];
csp.band= [7 30];

for dd= 1:length(dir_list),
  dir_name= dir_list{dd};
  is= find(dir_name=='_');
  subject= dir_name(1:is-1);
  file= strcat(dir_name, '/', {'imag_move', 'imag_lett'}, subject);
  [cnt, mrk]= eegfile_loadMatlab(file, 'clab',csp.clab);
  mrk= mrk_selectClasses(mrk, classes);
  
  [b, a]= butter(5, csp.band/cnt.fs*2);
  cnt_flt= proc_filt(cnt, b, a);
  fv= cntToEpo(cnt_flt, mrk, csp.ival);
  
  proc= struct('memo', 'csp_w');
  proc.train= ['[fv,csp_w]= proc_csp_auto(fv, 3); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
  proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
  xvalidation(fv, 'LDA', 'xTrials',[3 2], 'proc',proc);
end
