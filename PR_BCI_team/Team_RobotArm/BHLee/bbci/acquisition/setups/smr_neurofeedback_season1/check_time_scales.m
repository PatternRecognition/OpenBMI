subdir_list= get_session_list('leitstand_season2', 'check_TODAY_DIR', 0);

for vp= 1:length(subdir_list),
  
subdir= subdir_list{vp}
sbj= subdir(1:find(subdir=='_',1,'first')-1);
bbci= [];
%bbci.train_file= strcat(subdir, '/relax', sbj);
bbci.train_file= strcat(subdir, '/resting', sbj);

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

fig_set(1, 'gridsize',[2 1]);
clf;
hold on;

ilen_list= [500 750 1000 1500 2000 2500 3000 4000 5000 7500 10000];
col= cmap_rainbow(length(ilen_list));

for tt= 1:length(ilen_list),
  
fv= cntToEpo(cnt_flt, mrk_online, ilen_list(tt)*[-0.5 0.5], 'mtsp','after');
fv= proc_variance(fv);
fv= proc_logarithm(fv);
[fv, opt_smr]= proc_smr_extractor(fv);

if tt==1,
  T= length(fv.x);
end
plot([NaN*ones(1,(T-length(fv.x))/2) fv.x], '-', 'Color',col(tt,:));
set(gca, 'XLim',[0 T+1], 'YLim',[-0.2 1.2]);
end

legend(cprintf('%d ms', ilen_list), 'Location', 'EastOutside');
hold off;

printFigure(['/tmp/smr_online_time_scales_' sbj], [30 8]);

end
