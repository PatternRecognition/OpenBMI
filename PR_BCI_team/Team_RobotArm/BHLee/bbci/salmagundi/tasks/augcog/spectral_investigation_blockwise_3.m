setup_augcog;

augcog = augcog(13:18);

%% 'drive' must be first task -> used as baseline
tasks = {'carfollow','visual','auditory','calc'};

%% preliminary: the i-th cell specifies the frequency band for the
%% i-th experiment. it may also be a a nTasks x 2 matrix specifying
%% different frequency bands for each task in that experiment

band_list = {[8 12],...
             [8 12],...
             [5 12],...
             [8 14],...
             [7 12],...
             [8 12]};



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

classes = {'base*','low*','high*'};
nSubjects= length(augcog);
nTasks= length(tasks);
for ff= 1:nSubjects,
  bands= band_list{ff};
  if size(bands,1)==1,
    bands= repmat(bands, [nTasks 1]);
  end
  
  for tt= 1:nTasks,
    band= bands(tt,:);

    %% get all blocks -> blk structure
    blk = getAugCogBlocks(augcog(ff).file);
    %%  select only one task ('*' is for 'low ' resp. 'high ')
    blk = blk_selectBlocks(blk, ['*' tasks{tt}]);
% $$$     if isempty(blk.className),
% $$$       continue;
% $$$     end
% $$$     %%  and load the concatenated blocks
    if isempty(blk.y)
      continue;
    end
    [cnt,bmrk] = readBlocks(augcog(ff).file, blk);
      
    %% sort classes for convenience (low condition always first)
%    bmrk= mrk_selectClasses(bmrk, 'low*','high*');
    cnt= proc_selectChannels(cnt, 'not', 'E*','M*', 'Fp*', 'F7,8');
    
    %% determine 2d electrode positions for scalp plots
    mnt= projectElectrodePositions(cnt.clab);
    %% load a channel layout for grid plots
    mnt= setDisplayMontage(mnt, 'augcog_bipolar');
    
    bmrk = separate_markers(bmrk);
    %% generate markers each second within the given blocks
    mrk= mrk_evenlyInBlocks(bmrk, 1000);
    %% cut out epochs of 1 second following each marker
    epo= makeEpochs(cnt, mrk, [0 990]);
    
    %% artifact rejection
    %% (criterium: max - min value in one of the given exceeds 100 uV)
    crit.maxmin=150;
    iArte= find_artifacts(epo, {'F3,z,4','C3,z,4','P3,z,4'}, crit);
    fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
            length(iArte), crit.maxmin);
    epo= proc_removeEpochs(epo, iArte);

    %% calculate spectra of each epoch
    spec= proc_fourierBandMagnitude(epo, [3 35], hamming(epo.fs));

% $$$     if tt==1,  %% remember 'low drive' -> baseline for other conditions
% $$$       colorOrder= [0 0 0; 1 0 0];
% $$$       figSize= [9 7];
% $$$       spec_baseline= proc_selectClasses(spec, 'low drive*');
% $$$       spec_baseline.className= {'baseline'};
% $$$     else       %% add the baseline condition
      figSize= [17 7];
      colorOrder= [0 0 0; 0 0.7 0; 1 0 0;0 0 0.8;1 1 0; 1 0 1; 0 1 1; 0.8,0.8,0.8];
%      spec= proc_appendEpochs(spec_baseline, spec);
%    end

for i = 1:3
  indi{i} = getClassIndices(spec,classes{i});
  nindi(i) = length(indi{i});
end

ind = max(nindi);
classis = {};
for i = 1:3
  classis = cat(2,classis,repmat({classes{i}(1:end-1)},[1,ind-nindi(i)]));
  indi{i} = [indi{i},sum(nindi)+1:sum(nindi)+ind-nindi(i)];
  nindi(i) = ind;
end

spec.x = cat(3,spec.x,zeros(size(spec.x,1),size(spec.x,2),sum(nindi)-size(spec.y,1)));
spec.className = cat(2,spec.className,classis);
spec.y = cat(1,cat(2,spec.y,zeros(size(spec.y,1),sum(nindi)-size(spec.y,1))),cat(2,zeros(sum(nindi)-size(spec.y,1),size(spec.y,2)),eye(sum(nindi)-size(spec.y,1))));


ind = [indi{:}];

spec.y = spec.y(ind,:);
spec.className = spec.className(ind);

for i = 1:3
  spe = proc_selectClasses(spec,classes{i});

    grid_plot(spe, mnt, grid_opt, 'colorOrder',colorOrder);
    grid_markIval(band);
    saveFigure([fig_dir augcog(ff).file '_fft_' tasks{tt} '_blockwise_' classes{i}(1:end-1)], [20 18]);
end
    %% extract the area under the spectral peak in the specified frequency band
    
    for i = 1:length(spec.className)
      if ~strcmp(spec.className{i}(end-1:end),' I')
        c = strfind(spec.className{i},' ');
        if isempty(c)
          spec.className{i} = '';
        else
          spec.className{i} = spec.className{i}(c(end)+1:end);
        end
      end
    end
    peak= proc_peakArea(spec, band);
    
    peak.arrangement = [3,size(peak.y,1)/3];
    
    plotClassTopographies(peak, mnt, 0, scalp_opt, ...
                          'titleAppendix',sprintf('[%d %d] Hz', band));
    saveFigure([fig_dir augcog(ff).file '_fft_peak_' tasks{tt} '_blockwise'], [20 18]);

  end
end
