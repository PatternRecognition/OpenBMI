addpath([BCI_DIR 'acquisition/setups/labrotation2010_DmitryZarubin'])
subdir_list= get_session_list('leitstand_season2');
%subdir_list= get_session_list('season10');

clear y1 y2
unix('rm /tmp/bbci_smr_extractor_log/*.log');

for vp= 1:length(subdir_list),
  
subdir= subdir_list{vp}
sbj= subdir(1:find(subdir=='_',1,'first')-1);
bbci= [];
bbci.train_file= strcat(subdir, '/relax', sbj);

global TODAY_DIR
TODAY_DIR= '/tmp/';
bbci.setup= 'smr_extractor';
bbci.func_mrk= 'durchrauschen';
bbci.save_name= strcat(TODAY_DIR, 'bbci_smr_extractor');
bbci.feedback= '';
bbci_bet_prepare
mrk_orig= mrk;
bbci_bet_analyze
bbci_bet_finish

%% -- dry feedback run (classifier output only)
bbci_bet_apply_offline(cnt, mrk_orig, 'setup_list',bbci.save_name);

dd= dir('/tmp/bbci_smr_extractor_log/*.log');
outfile= sprintf('/tmp/fb_%s.log', sbj);
cmd= sprintf('grep Send /tmp/bbci_smr_extractor_log/%s > %s', dd.name, outfile);
unix(cmd);
cmd= sprintf('rm /tmp/bbci_smr_extractor_log/%s', dd.name);
unix(cmd);

[ts,y1{vp},y2{vp}]= textread(outfile, 'Send to udp at timestamp %d: [%f,%f]');

end

save([DATA_DIR 'results/smr_neurofeedback/online_check'], ...
     'subdir_list','y1','y2');
