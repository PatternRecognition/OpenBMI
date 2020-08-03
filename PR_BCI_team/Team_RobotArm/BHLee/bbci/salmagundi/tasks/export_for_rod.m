file= 'Matthias_06_02_09/imag_lettMatthias';
file_fb= 'Matthias_06_02_09/imag_fb1drnocursMatthias';


%% get original parameter settings
bbci= eegfile_loadMatlab(file_fb, 'vars','bbci');
csp_ival= bbci.setup_opts.ival;
filt_b= bbci.analyze.csp_b;
filt_a= bbci.analyze.csp_a;
csp_w_orig= bbci.analyze.csp_w;
csp_clab= bbci.setup_opts.clab;

%% train classifier on calibration data
[cnt, mrk, mnt]= eegfile_loadMatlab(file);
mrk= mrk_selectClasses(mrk, bbci.classes);
cnt= proc_selectChannels(cnt, csp_clab);
cnt= proc_linearDerivation(cnt, csp_w_orig, 'prependix','csp');
cnt= proc_filt(cnt, filt_b, filt_a);
fv= makeEpochs(cnt, mrk, csp_ival);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
C= trainClassifier(fv, 'MSR');


winlen_list= [100 200 500 750 1000 1500];

[cnt, mrk, mnt]= eegfile_loadMatlab(file_fb);

cnt= proc_selectChannels(cnt, csp_clab);
cnt= proc_linearDerivation(cnt, csp_w_orig, 'prependix','csp');

for ww= 1:length(winlen_list),
  ctrl= proc_movingVariance(cnt, winlen_list(ww));
  ctrl= proc_logarithm(ctrl);
  ctrl= proc_linearDerivation(ctrl, C.w);
  ctrl.clab= {sprintf('ctrl_%d', winlen_list(ww))};
  ctrl.x= ctrl.x + C.b;
  if ww==1,
    ctrlms= ctrl;
  else
    ctrlms= proc_appendChannels(ctrlms, ctrl);
  end
end

save([DATA_DIR 'eegExport/MK060209_multiscale'],'ctrlms','mrk');
