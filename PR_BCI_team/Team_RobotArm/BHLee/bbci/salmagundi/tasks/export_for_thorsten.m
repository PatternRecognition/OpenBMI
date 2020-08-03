subdir_list= {'VPjs_08_07_09', 'VPjx_08_07_16', 'VPkl_08_08_18', 'VPks_08_09_03', 'VPtae_08_06_17', 'VPkg_08_08_07', 'VPkp_08_08_27'};


for vp= 1:length(subdir_list),

  subdir= subdir_list{vp};  
  sbj= subdir(1:find(subdir=='_',1,'first')-1);

  file_cb= [subdir '/imag_arrow' sbj];
  file_fb= [subdir '/imag_fbarrow' sbj];
  bbci= eegfile_loadMatlab(file_fb, 'vars','bbci');
  
  classes= bbci.classes;
  csp_clab= bbci.analyze.clab;
  filt_b= bbci.analyze.csp_b;
  filt_a= bbci.analyze.csp_a;
  csp_w_orig= bbci.analyze.csp_w;
  csp_ival= bbci.analyze.ival;
  rejected_trials= bbci.analyze.rej_trials;

  if ~isequal(classes, {'left','right'}),
    error('wrong class combination');
  end
  
%  cnt= eegfile_readBV([file_cb '*'], 'fs',bbci.fs, ...
%                      'subsample_fcn','subsampleByLag', 'clab',csp_clab);
  [cnt, mrk]= eegfile_loadMatlab(file_cb, 'clab',csp_clab);
  mrk= mrk_chooseEvents(mrk, 'not', bbci.analyze.rej_trials);
  mrk= mrk_selectClasses(mrk, classes);
  
  cnt= proc_filt(cnt, filt_b, filt_a);
  epo_csp_calib= proc_linearDerivation(cnt, csp_w_orig);
  epo_csp_calib= cntToEpo(epo_csp_calib, mrk, csp_ival);
  fv_calib= proc_variance(epo_csp_calib);
  fv_calib= proc_logarithm(fv_calib);
  cls_calib= applyClassifier(fv_calib, 'LSR', bbci.cls.C);
  err= loss_rocArea(fv_calib.y, cls_calib);
  fprintf('%s - calib err: %.1f\n', sbj, 100*err);
  
  %% Load feedback data
  [cnt, mrk]= eegfile_loadMatlab(file_fb, 'clab',csp_clab);
  cnt= proc_filt(cnt, filt_b, filt_a);
  epo_csp_feedb= cntToEpo(cnt, mrk, csp_ival);
  epo_csp_feedb= proc_linearDerivation(epo_csp_feedb, csp_w_orig);
  fv_feedb= proc_variance(epo_csp_feedb);
  fv_feedb= proc_logarithm(fv_feedb);
  cls_feedb= applyClassifier(fv_feedb, 'LSR', bbci.cls.C);
  err= loss_rocArea(fv_feedb.y, cls_feedb);
  fprintf('%s - feedb err: %.1f\n', sbj, 100*err);

  save_name= [DATA_DIR 'eegExport/' sbj '_csp_epochs'];
  save(save_name, ...
       'epo_csp_calib', 'fv_calib', 'cls_calib', ...
       'epo_csp_feedb', 'fv_feedb', 'cls_feedb');
end  %% for vp
