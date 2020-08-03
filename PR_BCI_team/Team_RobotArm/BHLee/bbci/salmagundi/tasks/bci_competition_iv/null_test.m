compet_dir= [DATA_DIR 'bci_competition_iv/'];

sbj_code= 'abfg';
sbj_code= 'abfgcde';
for si= 1:length(sbj_code),
  sbj= sbj_code(si);
  fprintf('Subject %s\n', sbj);
  
  %% get true labels
  load([compet_dir 'BCICIV_eval_ds1' sbj '_1000Hz_true_y'], 'true_y');
  tt= floor(length(true_y)/10);
  true_y100= mean(reshape(true_y(1:10*tt), [10 tt]), 1)';
  
  zerr(si)= nanmean((true_y100-zeros(size(true_y100))).^2);
end


return


T_crop= 1759140;
sbj_code= 'abfg';
for k= 1:length(sbj_code),
  sbj= sbj_code(k);
  
  %% get true labels
  load([compet_dir 'BCICIV_eval_ds1' sbj '_1000Hz_true_y'], 'true_y');
  if k==1,
    true_y= true_y(1:T_crop);
  end
  
  zerr(k)= nanmean((true_y-zeros(size(true_y))).^2);
  fprintf('Subject %s:  %.3f\n', sbj, zerr(k));
end
