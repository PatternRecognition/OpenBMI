file = {'VPar_031205','VPkm_031201','VPth_031210','VPts_031209', ...
	'VPum_031203'};

features = {'*drive','*carfollow','*visual','*auditory','*calc'};

fig_dir = 'augcog_misc';

for i = 1:length(file);
  for j = 1:length(features);
    [blk,mnt] = getAugCogBlocks(file{i});
    blk = blk_selectBlocks(blk,features{j});
    [cnt,mrk] = readBlocks(blk);
    po = diff([mrk.pos,mrk.end]);
    po = min(po)-10;
    po = po*10;
    epo = makeEpochs(cnt,mrk,[0 po]);
    epo.title = sprintf('%s_%s_laplace_large',file{i},features{j}(2:end));
    fv = proc_laplace(epo,'large',[],'filter all');
    fv = proc_spectrum(fv,[2 45],fv.fs);
    grid_plot(fv,mnt);
    saveFigure([fig_dir '/' file{i} '_spectrum_large_' features{j}(2:end)],[20 ...
		    20]);
    epo.title = sprintf('%s_%s_laplace_diagonal',file{i},features{j}(2:end));
    fv = proc_laplace(epo,'diagonal',[],'filter all');
    fv = proc_spectrum(fv,[2 45],fv.fs);
    grid_plot(fv,mnt);
    saveFigure([fig_dir '/' file{i} '_spectrum_diagonal_' features{j}(2:end)],[20 ...
		    20]);
    
    epo.title = sprintf('%s_%s_car',file{i},features{j}(2:end));
    fv = proc_commonAverageReference(epo);
     fv = proc_spectrum(fv,[2 45],fv.fs);
    grid_plot(fv,mnt);
    saveFigure([fig_dir '/' file{i} '_spectrum_car_' features{j}(2:end)],[20 ...
		    20]);
   
  end
end

