setup_augcog;
tasks = {'drive','carfollow','visual','auditory','calc','comb'};
band_list= {[8 12],          [7 12], [8 11], [8 12], [9 13], ...
            [8 12], [14 19], [7 12], [8 11], [8 12], [9 13]};

fig_dir = 'augcog_misc/';
grid_opt= struct('colorOrder', [0 0.7 0; 1 0 0]);
%grid_opt.scaleGroup= {scalpChannels, {'EOG*'}, {'EMG'}, {'Ekg*'}};
%grid_opt.scalePolicy= 'auto';
grid_opt.axisTitleFontWeight= 'bold';
grid_opt.axisTitleHorizontalAlignment= 'center';
scalp_opt= struct('shading','flat', 'resolution',20, 'contour',-5);

for ff= 1:length(file),
  band= band_list{ff};
  for tt= 1:length(tasks),
    blk= getAugCogBlocks(augcog(ff).file);
    blk= blk_selectBlocks(blk, ['*' tasks{tt}]);
    if isempty(blk.className),
      continue;
    end
    [cnt,mrk]= readBlocks(augcog(ff).file, blk);
    %% sort classes
    mrk= mrk_selectClasses(mrk, 'low*','high*');
%    bip= proc_bipolarChannels(cnt, 'Fp2-Eog', 'F7-F8', 'MT1-MT2');
%    bip.clab= {'EOGv','EOGh','EMG'};
%    cnt= proc_appendChannels(cnt, bip);
%    cnt= proc_removeChannels(cnt, 'Eog','MT1','MT2');
    cnt= proc_selectChannels(cnt, 'not', 'E*','M*', 'Fp*');

    mnt= projectElectrodePositions(cnt.clab);
    mnt= setDisplayMontage(mnt, 'augcog_bipolar');
%    mnt= excenterNonEEGchans(mnt);
    po = diff([mrk.pos,mrk.end]);
    po = min(po)-10;
    po = po*1000/mrk.fs;
    epo = makeEpochs(cnt, mrk, [0 po]);
%    fv = proc_laplace(epo, 'diagonal', ' lap', 'filter all');
%    fv = proc_spectrum(fv, [2 45], fv.fs);
%    grid_plot(fv, mnt, grid_opt);
%    saveFigure([fig_dir augcog(ff).file '_spectrum_diagonal_' tasks{tt}], ...
%               [20 16]);

    fv = proc_commonAverageReference(epo, {'not','Fp*'});
    fv = proc_spectrum(fv, [5 35], fv.fs);
    grid_plot(fv, mnt, grid_opt);
    grid_markIval(band);
    saveFigure([fig_dir augcog(ff).file '_spectrum_car_' tasks{tt}], [20 16]);
    
    fv = proc_spectrum(epo, [5 35], epo.fs);
    plotClassTopographies(fv, mnt, [30 35], scalp_opt);
    saveFigure([fig_dir augcog(ff).file '_spectrum_noise_' tasks{tt}], [15 7]);
    
    plotClassTopographies(fv, mnt, band, scalp_opt);
    saveFigure([fig_dir augcog(ff).file '_spectrum_scalp_' tasks{tt}], [15 7]);

    sp= proc_peakArea(fv, band);
    plotClassTopographies(sp, mnt, 0, scalp_opt, ...
                          'titleAppendix',sprintf('[%d %d] Hz', band));
    saveFigure([fig_dir augcog(ff).file '_spectrum_peak_' tasks{tt}], [15 7]);
  end
end
