subdir_list= textread([BCI_DIR 'studies/season2/session_list_ext'], '%s');

for vp= 1:length(subdir_list),
  sub_dir= [subdir_list{vp} '/'];
  is= min(find(sub_dir=='_'));
  sbj= sub_dir(1:is-1);

  %% load data of calibration (training) session
  if strcmp(sbj, 'VPco'),
    [cnt,mrk,mnt]= eegfile_loadMatlab([sub_dir 'real' sbj]);
  else
    [cnt,mrk,mnt]= eegfile_loadMatlab([sub_dir 'imag_lett' sbj]);
  end
  disp([sub_dir 'imag_lett' sbj])
  
  %% get the two classes that have been used for feedback
  bbci= eegfile_loadMatlab([sub_dir 'imag_1drfb' sbj], 'vars','bbci');
  classes= bbci.classes;
  mrk= mrk_selectClasses(mrk, classes);
  mnt = setDisplayMontage(mnt, 'small');
  cnt = proc_selectChannels(cnt, cnt.clab{find(~isnan(mnt.box(1, 1:end-1)))});

  epo = makeEpochs(cnt, mrk, [0 3500]);

  epo = proc_specgram(epo);
  epo = proc_r_square_signed(epo);
  grid_imagesc(epo, mnt);
  
  pause
end
