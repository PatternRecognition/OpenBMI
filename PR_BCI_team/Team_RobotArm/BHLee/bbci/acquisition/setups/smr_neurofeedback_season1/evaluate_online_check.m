load([DATA_DIR 'results/smr_neurofeedback/online_check']);


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

close all

mrk_online= mrk_evenlyInBlocks(blkcnt, 40);
fv= cntToEpo(cnt_flt, mrk_online, [-opt.ilen_apply(1) 0], 'mtsp','before');
fv= proc_variance(fv);
fv= proc_logarithm(fv);
[fv, opt_smr]= proc_smr_extractor(fv);
ff= cntToEpo(cnt_flt, mrk_online, [-opt.ilen_apply(2) 0], 'mtsp','before');
ff= proc_variance(ff);
ff= proc_logarithm(ff);
ff= proc_smr_extractor(ff, opt_smr);

fig_set(1);
T= length(y1{vp});
plot([NaN*ones(1,T-length(fv.x)) fv.x], 'ro');
set(gca, 'XLim',[0 length(fv.x)+1], 'YLim',[-0.2 1.2]);
hold on
plot([NaN*ones(1,T-length(ff.x)) ff.x], 'ko');
legend(cprintf('%d ms', opt.ilen_apply));
plot(y1{vp}, 'rx');
plot(y2{vp}, 'kx');
hold off;

pause

end
