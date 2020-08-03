subdir_list= textread([BCI_DIR 'studies/season3/session_list'], '%s');
td= '/home/neuro/data/dropbox/blanker/i_wanna_be_burned_by_you';
mkdir(td)

for vp= 1:length(subdir_list),

  sd= subdir_list{vp}
  cmd= sprintf('mkdir %s/%s', td, sd);
  [s,w]= unix(cmd);
  if s,
    fprintf('error: %s\n', w);
  end
  cmd= sprintf('cp %s/%s/imag_lett* %s/%s', EEG_MAT_DIR, sd, td, sd);
  [s,w]= unix(cmd);
  if s,
    fprintf('error: %s\n', w);
  end
  cmd= sprintf('cp %s/%s/imag_1drfb* %s/%s', EEG_MAT_DIR, sd, td, sd);
  [s,w]= unix(cmd);
  if s,
    fprintf('error: %s\n', w);
  end
  cmd= sprintf('cp %s/%s/arte* %s/%s', EEG_MAT_DIR, sd, td, sd);
  [s,w]= unix(cmd);
  if s,
    fprintf('error: %s\n', w);
  end
end
