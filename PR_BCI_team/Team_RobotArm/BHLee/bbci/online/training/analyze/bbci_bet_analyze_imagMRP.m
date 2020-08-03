grid_opt.colorOrder= [1 0 0; 0 0.7 0; 0 0 1; 0.9 0.8 0;0,0.8,0.9];
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOGh'}, {'EOGv'}};
grid_opt.scalePolicy= {'auto', [-10 10], 'sym', 'auto'};

rsqu_opt= {'colorOrder',[0.9 0.9 0; 1 0 1; 0 0.8 0.8], ...
           'scalePolicy','auto'};
grd= sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
mnt_lap= setDisplayMontage(mnt, grd);

disp_clab= getClabOfGrid(mnt);
requ_clab= getClabForLaplace(Cnt, disp_clab);
fig_opt= {'numberTitle','off', 'menuBar','none'};

if bbci.withgraphics
  epo = proc_selectChannels(Cnt,requ_clab);
  epo= makeEpochs(epo, mrk, opt.dar_ival);
  erp= proc_subtractMovingAverage(epo, opt.dar_sma, opt.dar_causal);
  erp= proc_movingAverage(erp, opt.dar_ma, opt.dar_causal);
  erp  = proc_laplace(erp);

  erp= proc_baseline(erp, opt.dar_baseline);
  if length(bbci.classes)==2
    erp_rsq= proc_r_square(proc_selectClasses(erp,bbci.classes));
  end

  handlefigures('use','ERP');
  set(gcf, fig_opt{:},  ...
           'name',sprintf('%s: ERP in [%d %d] ms', Cnt.title, opt.dar_ival));


  h = grid_plot(erp, mnt_lap, grid_opt);
  grid_markIval([opt.ival(1),opt.ival(end)]);
  if length(bbci.classes)==2
    hh = hot(64); hh = hh(end:-1:1,:);
    hh(1,:) = [0.999999,0.999999,0.999999];
    colormap(hh)
    grid_addBars(erp_rsq,'colormap',hh,'height',0.12,'h_scale',h.scale);
  end


  head= mnt;
  scalp_opt= struct('shading','flat', 'resolution',20, 'contour',-4);
  handlefigures('use','ERPscalps');
  set(gcf, fig_opt{:},  ...
           'name',sprintf('%s:  Patterns', Cnt.title));
  scalpEvolution(epo, head, opt.dar_scalps, scalp_opt);

end



cnt = proc_selectChannels(Cnt,opt.clab);
cnt = proc_subtractMovingAverage(cnt,opt.sma,opt.causal);
cnt = proc_movingAverage(cnt,opt.ma,opt.causal);
mrk_cl = mrk_selectClasses(mrk,bbci.classes{:});
for i = 1:length(opt.ival)-1
  epo = makeEpochs(cnt,mrk_cl,[opt.ival(i),opt.ival(i+1)]);
  epo = proc_jumpingMeans(epo,opt.jMeans);
  for j = 1:length(epo.className)
    epo.className{j} = [epo.className{j} ' ' int2str(i)];
  end
  if ~isfield(epo,'bidx')
    epo.bidx = 1:size(epo.y,2);
  end

  if i==1
    features = epo;
  else
    features = proc_appendEpochs(features,epo);
  end
end



if length(opt.ival)>2
  features = proc_combineClasses(features,features.className(1:2:size(features.y,1)),features.className(2:2:size(features.y,1)));
end


bbci_bet_message('Outlierness\n');
fig1 = handlefigures('use','trial-outlierness');
set(gcf, fig_opt{:}, ...
         'name',sprintf('%s: Sorted Variances', Cnt.title));
fig2 = handlefigures('use','channel-outlierness');
set(gcf, fig_opt{:}, ...
         'name',sprintf('%s: 2D Plot', Cnt.title));


if isfield(opt, 'threshold') & opt.threshold<inf
  features = proc_outl_slow(features,struct('trialthresh',opt.threshold,...
                               'display',bbci.withclassification,...
                               'handles',[fig1 fig2]));
else
  proc_outl_slow(features,struct('display',bbci.withclassification,...
                          'handles',[fig1 fig2]));
end



if bbci.withclassification
  class_opt = struct('outer_ms',1,'msTrials',[2 10 -1],'xTrials',[5 10]);
  [loss,loss_std] = xvalidation(features,opt.model,class_opt);
  remainmessage = sprintf('Classification: %2.1f +/- %1.1f\nInside Classification: ',100*loss,100*loss_std);
  bbci_bet_message('Classification: %2.1f +/- %1.1f\n',100*loss,100*loss_std);

end

% What do we need later?
analyze = struct('features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
