setup_augcog;

%% use only tasks with alternating blocks here
tasks = {'auditory', 'calc', 'visual'};

%% preliminary: the i-th cell specifies the frequency band for the
%% i-th experiment. it may also be a a nTasks x 2 matrix specifying
%% different frequency bands for each task in that experiment
band_list= {[7 12], ...
            [8 12], ...
            [8 11], ...
            [8 12], ...
            [9 13], ...
            [7 12], ...
            [8 12], ...
            [8 12], ...
            [8 11], ...
            [8 12], ...
            [9 13]};

%% sub folder in which figures are saved
%%  (if it is a relative path, it is appended to the global variable TEX_DIR)
fig_dir = 'augcog_misc/';

%% default options for grid plots
grid_opt.axisTitleFontWeight= 'bold';
grid_opt.axisTitleHorizontalAlignment= 'center';
grid_opt.colorOrder= [0 0.7 0; 1 0 0; 0 0 1];
%%  and displaying scalp topographies
scalp_opt= struct('shading','flat', 'resolution',20, 'contour',-5);
%% shading 'interp' gives smoother plots, eps file are getting big
%% bigger resolution produces -- "" --
%% contour -5 means try to place approx. 5 contour lines in the
%%  given value range, but try to find 'nice' values

nTasks= length(tasks);
for ff= 6:11,
  bands= band_list{ff};
  if size(bands,1)==1,
    bands= repmat(bands, [nTasks 1]);
  end
  
  for tt= 1:nTasks,
    band= bands(tt,:);
    
    %% get all blocks -> blk structure
    blk = getAugCogBlocks(augcog(ff).file);
    %%  and select only one task ('*' is for 'low ' resp. 'high ')
    blk = blk_selectBlocks(blk, ['*' tasks{tt}]);
    if isempty(blk.className),
      continue;
    end
    
    %% define one big block encompassing all individual blocks
    blk_all= struct('fs',blk.fs, 'y',1, 'className',{tasks(tt)}, ...
                    'ival',blk.ival([1;end]));
    %%  and load them
    cnt= readBlocks(augcog(ff).file, blk_all);
    %% shift individual blocks to fit for the loaded segment
    bmrk= blk;
    bmrk.ival= bmrk.ival - blk.ival(1) + 1;
    bmrk.pos= bmrk.ival(1,:);
    bmrk= blk_addInterimBlocks(bmrk);

    %% sort classes for convenience (low condition always first)
    bmrk= blk_selectBlocks(bmrk, 'low*','high*');
    %% exclude some non-relevant channels
    cnt= proc_selectChannels(cnt, 'not', 'E*','M*', 'Fp*', 'F7,8');
    
    %% determine 2d electrode positions for scalp plots
    mnt= projectElectrodePositions(cnt.clab);
    %% load a channel layout for grid plots
    mnt= setDisplayMontage(mnt, 'augcog_bipolar');
    
    %% generate markers each second
    mrk= mrk_evenlyInBlocks(bmrk, 1000);
    %% cut out epochs of 1 second following each marker
    epo= makeEpochs(cnt, mrk, [0 990]);
    
    %% artifact rejection
    %% (criterium: max - min value in one of the given exceeds 100 uV)
    crit.maxmin= 100;
    iArte= find_artifacts(epo, {'F3,z,4','C3,z,4','P3,z,4'}, crit);
    fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
            length(iArte), crit.maxmin);
    epo= proc_removeEpochs(epo, iArte);
    
    %% calculate spectra of each epoch
    spec= proc_fourierBandMagnitude(epo, [3 35], hamming(epo.fs));
    %% extract the area under the spectral peak in the specified frequency band
    peak= proc_peakArea(spec, band);

    %% select channels that reflect conditional peak area modulation:
    %%  exclude temporal channels
    rsq= proc_selectChannels(peak, 'not', 'T*');
    %%  and calculate the r^2 difference value for each channel
    rsq= proc_r_square(rsq);
    %%  find all channels that have more than 85% of the maximal r^2 value
    [so,si]= sort(rsq.x);
    ii= min(find(so>0.85*so(end)));
    %%  and select at most 5
    ii= max(ii, length(si)-4);
    good_chans= rsq.clab(si(ii:end));
    fv= proc_selectChannels(peak, good_chans);
    nChans= length(good_chans);
    %% average peak values for all good channels
    fv= proc_linearDerivation(fv, ones(nChans,1)/nChans);
    
    clf;
    color_list= [0 0.7 0; 1 0 0; 0 0 1];
    ap= squeeze(fv.x);
    ap_ma= movingAverage(ap, 30);
    for cc= 1:size(fv.y,1),
      idx= find(fv.y(cc,:));
      plot(idx, ap(idx), '.', 'color',color_list(cc,:));
      hold on
    end
    for cc= 1:size(fv.y,1),
      idx= find(fv.y(cc,:));
      tmp= NaN*zeros(size(ap));
      tmp(idx)= ap_ma(idx);
      plot(tmp, 'linewidth',3, 'color',color_list(cc,:));
    end
    hold off
    legend(fv.className);
    xlabel('time [s]');
    ylabel('\alpha peak value');
    title(untex(augcog(ff).file));
    
%   saveFigure([fig_dir augcog(ff).file '_alpha_peak_temporal_' tasks{tt}], ...
%               [15 7]);

  end
end
