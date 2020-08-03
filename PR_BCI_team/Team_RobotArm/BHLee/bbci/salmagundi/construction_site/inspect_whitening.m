for vp= setdiff(1:length(subdir_list), 5),

  sub_dir= [subdir_list{vp} '/'];
  is= min(find(sub_dir=='_'));
  sbj= sub_dir(1:is-1);

  [cnt,mrk,mnt]= eegfile_loadMatlab([sub_dir 'imag_lett' sbj]);
  bbci= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','bbci');
  mrk= mrk_selectClasses(mrk, bbci.classes);
  cnt= proc_selectChannels(cnt, 'not','E*');  %% do NOT use EMG, EOG
  cnt= proc_selectChannels(cnt, getClabForLaplace(cnt, clab));
  band= select_bandnarrow(cnt, mrk, test_ival);
  [filt_b,filt_a]= butter(5, band/cnt.fs*2);
  cnt= proc_filt(cnt, filt_b, filt_a);

figure(1);
  fv= makeEpochs(cnt, mrk, test_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  scalpPatterns(fv, mt, []);

figure(2)
  [cnt_lap,lap_w]= proc_laplace(cnt);
  fv= makeEpochs(cnt_lap, mrk, test_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  scalpPatterns(fv, mt, []);

figure(3);
  [cnt_w]= proc_whiten(cnt);
  cnt_w.clab= cnt.clab;
  fv= makeEpochs(cnt_w, mrk, test_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  scalpPatterns(fv, mt, []);

figure(4);
  [cnt_w_lap]= proc_laplace(cnt_w);
  fv= makeEpochs(cnt_w_lap, mrk, test_ival);
  fv= proc_variance(fv);
  fv= proc_logarithm(fv);
  scalpPatterns(fv, mt, []);
  drawnow;
  
end
