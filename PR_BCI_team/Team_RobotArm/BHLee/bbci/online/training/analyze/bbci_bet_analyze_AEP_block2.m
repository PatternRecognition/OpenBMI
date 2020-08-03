grd= sprintf('EOGh,scale,Fz,legend,EOGv\nC3,C1,Cz,C2,C4\nP3,P1,Pz,P2,P4\nEMGl,O1,EMGf,O2,EMGr');

mrk_m = mrk;
mrk = proc_combineClassesKeyword(mrk, 'st', 'tar', 'name1', 'Non-target', 'name2', 'Target'); 
mrk.y(find(mrk.y)) = 1;
mnt= getElectrodePositions(Cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

fig_opt= {'numberTitle','off', 'menuBar','none'};

scalp_opt= struct('shading','flat', 'resolution',30, 'contour',[-40:2:40]);
head= setDisplayMontage(mnt, 'visible_128');
head= mnt_adaptMontage(head, Cnt.clab);
colDef= {'Target','Non-target';
         [0 0.7 0], [1 0 0]};
grid_opt = defopt_erps;
grid_opt.colorOrder = choose_colors(mrk.className,colDef);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};
scalp_opt.colorOrder = grid_opt.colorOrder;

handlefigures('use','ARTIFACT');
% [mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 800], 'visualize', 1, 'do_multipass', 1, 'do_bandpass', 0);
[mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 800], 'visualize', 1, 'do_multipass', 1, 'do_bandpass', 0, 'whiskerlength', 4);
Cnt = proc_selectChannels(Cnt, 'not', rclab);
epo = cntToEpo(Cnt,mrk_rej,opt.ival);
%clear Cnt;
epo = proc_baseline(epo,opt.baseline, 'beginning_exact');
epo  = proc_selectChannels(epo, 'not', {'E*', 'Mas*'});
epo_r= proc_r_square_signed(proc_selectClasses(epo,'Target', 'Non-target'));
% opt.selectival = select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',3, 'visualize', 0, 'sort', 1);

opt.selectival = [110 170; 220 290; 300 380];
%opt.selectival = [90 150; 150 210; 240 370 ; 550 600];

% opt.selectival = [90 200; 260 380];
% opt.selectival = [270 370; 430 590]; 380 420;
%opt.selectival = input('Give intervals: ');
features= proc_jumpingMeans(epo, opt.selectival);

% There's a problem here with proc_flaten
[features, opt.meanOpt] = proc_subtractMean(proc_flaten(features));
[features, opt.normOpt] = proc_normalize(features);

if bbci.withgraphics
  handlefigures('use','INTERVAL');
  select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), 'nIvals',3, 'visualize', 1);
  
  epo= proc_baseline(epo, opt.dar_base, 'beginning_exact');
  epo_r= proc_r_square_signed(proc_selectClasses(epo,'Target', 'Non-target'));
  
  handlefigures('use','ERP');
  set(gcf, fig_opt{:}, 'name',sprintf('%s: ERPs in [%d %d] ms', epo.title, opt.dar_ival));  
  h = grid_plot(epo, mnt, grid_opt);
  
  hh = cmap_posneg(81);
  colormap(hh)
  grid_markIval(opt.selectival);
  grid_addBars(epo_r,...
               'colormap',hh,'height',0.12,'h_scale',h.scale);
 
  handlefigures('use','ERPscalps');
  set(gcf, fig_opt{:}, ...
           'name',sprintf('%s: ERP-Pattern', epo.title));
  scalpEvolutionPlusChannel(epo, head, 'Cz', opt.selectival, scalp_opt);
  grid_addBars(epo_r, 'colormap',hh,'height',0.12);

  handlefigures('use','ERP r-value scalps');  
  scalpEvolution(epo_r, head, opt.selectival, scalp_opt, 'colormap', cmap_posneg(81));
end

clear epo epo_r

if bbci.withclassification
  opt_xv= strukt('xTrials', [5 5], 'loss','rocArea');
  [dum,dum,outTe] = xvalidation(features, opt.model, opt_xv);
  me= val_confusionMatrix(features, outTe, 'mode','normalized');
  remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
  bbci_bet_message(remainmessage,0);
else
    remainmessage = '';
end

if bbci.withthreshold
  cls.C = trainClassifier(features,'FDshrink');
%   Cnt = proc_selectChannels(Cnt, 'not', rclab);
  epo = cntToEpo(Cnt,mrk,opt.ival);
  clear Cnt;
  epo = proc_baseline(epo,opt.baseline, 'beginning_exact');
  epo  = proc_selectChannels(epo, 'not', {'E*', 'Mas*'});
  fv= proc_jumpingMeans(epo, opt.selectival);
  fv = proc_subtractMean(proc_flaten(fv), opt.meanOpt);
  fv = proc_normalize(fv, opt.normOpt);
  out = apply_separatingHyperplane(cls.C, fv.x);
  out = reshape(out, 6, 15, []);
  [clasDir dum] = find(mrk_m.y);
  corrDir = clasDir(find(clasDir > 8));
  corrDir = corrDir(1:15:length(corrDir))-8;
  clasDir(find(clasDir > 8)) = clasDir(find(clasDir > 8)) - 8;
  clasDir = reshape(clasDir, 6,15,[]);
  for i = 1:size(out, 2),
    for j = 1:size(out, 3),
      out(clasDir(:, i, j), i, j) = out(:, i, j);
    end
  end
  
  corrScore = {};
  incorrScore = {};
  winner = zeros(15, size(out, 3));
  for itId = 1:15,
      indSc = squeeze(median(out(:,1:itId, :), 2));
      [dum winner(itId, :)] = min(indSc, [], 1);
      corrId = winner(itId,:) == corrDir';
      incorrId = ~corrId;

      srtSc = sort(indSc, 1);
      srtSc = srtSc(2,:)-srtSc(1,:);

      corrScore{itId} = srtSc(corrId);
      incorrScore{itId} = srtSc(incorrId);
  end
  meanCorrSc = cellfun(@percentiles, corrScore, num2cell(repmat(50, 1,15)));
  percCorrSc = cellfun(@percentiles, corrScore, repmat({[2 98]},1,15), 'UniformOutput', false);
  percCorrSc = reshape([percCorrSc{:}], 2, 15);
  
  if any(cellfun(@isempty, incorrScore)),
    for itI = 1:length(incorrScore),
      if isempty(incorrScore{itI}),
        meanInCorrSc(itI) = 0;
        percInCorrSc(:,itI) = [0; 0];      
      else
        meanInCorrSc(itI) = percentiles(incorrScore{itI}, 50);
        percInCorrSc(:,itI) = percentiles(incorrScore{itI}, [2 98]);
      end
    end
  else
    meanInCorrSc = cellfun(@percentiles, incorrScore, num2cell(repmat(50, 1,15)));
    percInCorrSc = cellfun(@percentiles, incorrScore, repmat({[2 98]},1,15), 'UniformOutput', false);
    percInCorrSc = reshape([percInCorrSc{:}], 2, 15);
  end
  
  x = [1:15];
  p = polyfit(x, percInCorrSc(2,:), 3);
  newy = polyval(p, x);
  earlyDec = [];
  clear label totAbovThres
  for i = 2:15,
      stopId = find(corrScore{i} > newy(i));
      newDec = ~ismember(stopId, earlyDec);
      totAbovThres(i) = length(find(newDec));   
      decTaken = winner(i,stopId(newDec));
      if isempty(decTaken),
        cDec = [];
      else
        cDec = decTaken == corrDir(stopId(newDec))';
      end
      label{i} = sprintf('%i\n%i / %i', totAbovThres(i), length(find(cDec)), length(find(~cDec)));
      earlyDec = [earlyDec stopId(newDec)];
  end  
  
  % do the plot
  handlefigures('use','THRESHOLD');
  hold off;
  col1 = [.5 0 1];
  col_pal1 = col_makePale(col1, .4);
  col2 = [.1 1 .6];
  col_pal2 = col_makePale(col2, .4);

  titFont = 20;
  normFont = 16;
  pl = plot(x, meanCorrSc', 'r', x, meanInCorrSc', 'b', 'LineWidth', 2);
  legend('Correct', 'Incorrect');
  set(pl(1), 'color', col_pal1);
  set(pl(2), 'color', col_pal2);
  set(gca, 'XLim', [.5 15.5]);
  % set(gca, 'YLim', [-.5 2.5]);
  pos = get(gca, 'Position');
  set(gca, 'FontSize', normFont);
  set(gca, 'Position', [pos(1), .2, pos(3) .65]);
  set(gca, 'tickDir', 'out');
  set(gcf, 'color', 'white');
  set(gca, 'Box', 'off');   
  hold on;
  % p1 = patch([x, fliplr(x)]', [meanCorrSc+sdCorSc, fliplr(meanCorrSc-sdCorSc)]', 'r');
  % p2 = patch([x, fliplr(x)]', [meanInCorrSc+sdInCorSc, fliplr(meanInCorrSc-sdInCorSc)]', 'b');
  p1 = patch([x, fliplr(x)]', [percCorrSc(1,:), fliplr(percCorrSc(2,:))]', 'r');
  p2 = patch([x, fliplr(x)]', [percInCorrSc(1,:), fliplr(percInCorrSc(2,:))]', 'b');
  t = text(x, mean([percCorrSc(2,:); percInCorrSc(2,:)]), label, 'HorizontalAlignment', 'center');

  plot(x, newy, 'k', 'Linewidth', 2);
  set([p1 p2], 'Edgecolor', 'none');
  set(p1, 'Facecolor', col_pal1);
  set(p2, 'Facecolor', col_pal2);
  set([p1 p2], 'FaceAlpha', .5);
  set(gca, 'XTick', [1:15]);
  set(gca, 'XTickLabel', [1:15]);

  xlabel('Nr of iterations');
  lab = ylabel('Diff between rank 1 and 2', 'FontSize', titFont);
  title('Rank diff per iteration', 'FontSize', titFont);
end

% What do we need later?
newy = max(newy, meanCorrSc);

thresholds = zeros(1,17);
thresholds(3:17) = newy;
analyze = struct('features', features, 'message', remainmessage, 'thresholds', abs(thresholds));

bbci_bet_message('Finished analysis\n');
