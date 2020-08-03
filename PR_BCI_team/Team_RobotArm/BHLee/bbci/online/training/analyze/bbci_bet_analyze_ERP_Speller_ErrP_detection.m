default_grd= sprintf(['scale,FC3,FC1,FCz,FC2,FC4,legend\n' ...
                     'C5,C3,C1,Cz,C2,C4,C6\n' ...
                     'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
                     'P7,P5,P3,Pz,P4,P6,P8\n' ...
                     'PO7,PO3,O1,Oz,O2,PO4,PO8']);

default_crit= struct('maxmin', 100, ...
                     'clab', 'EOG*', ...
                     'ival', [100 800]);
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'clab', '*', ...
                 'exclude_channels', {}, ...
                 'reject_artifacts', 0, ...
                 'reject_artifacts_opts', {}, ...
                 'reject_channels', 0, ...
                 'reject_eyemovements', 0, ...
                 'reject_eyemovements_crit', default_crit, ...
                 'grd', default_grd, ...
                 'clab_erp', {'CPz'}, ...
                 'clab_rsq', {'CPz','PO7'}, ...
                 'widely_nontarget', 0, ...
                 'withclassification', 1, ...
                 'nhist', 30, ...
                 'analyze_step', 1);

clear fv*
mnt= mnt_setGrid(mnt, opt.grd);
opt_scalp_erp= defopt_scalp_erp('colorOrder', [0.9 0 0.9; 0.4 0.4 0.4], ...
                                'extrapolate', 1, ...
                                'renderer', 'contourf', ...
                                'legend_pos','NorthWest');
opt_scalp_r= defopt_scalp_r('lineWidth', 2, ...
                            'channelAtBottom',1, ...
                            'extrapolate', 1, ...
                            'renderer', 'contourf', ...
                            'legend_pos','NorthWest');
opt_fig= {'numberTitle','off', 'menuBar','none'};


% remove exclude channels:
clab= Cnt.clab(chanind(Cnt, opt.clab));
opt.cfy_clab{aaa}= Cnt.clab(chanind(Cnt, opt.cfy_clab{aaa})); % dangerous: aaa might not yet be set.
if ~isempty(opt.exclude_channels)
  clab(strpatternmatch(opt.exclude_channels, clab)) = [];
  opt.cfy_clab{aaa}(strpatternmatch(opt.exclude_channels, opt.cfy_clab{aaa}))= [];
end

% select only the cfy_clab channels:
clab = clab_in_preserved_order(clab, opt.cfy_clab{aaa});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Select Intervals:     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ismember(opt.analyze_step, [1 2 5])
  aaa = opt.analyze_step;
  if isfield(bbci, 'analyze')
    analyze = bbci.analyze;
  end
    
  %% select and prepare marker structure:
  if numel(Cnt.T)>2
    warning('More than 2 trainingfiles used. Check carefully the use of Cnt.T in the following lines.');
  end
  if aaa==1
    % select only speller calibration data
    mk = mrk_selectEvents(mrk, mrk.pos<=Cnt.T(1));
    mk = mrk_selectEvents(mrk, mrk.trial_idx ~= 0);
  elseif aaa==5
    % ErrP classification on online spelling data: (trainfile must only
    % contain online spelling data)
    mk = mrk.classified;
    mk.y = [mk.error; ~mk.error];
    mk.className = {'error', 'correct'};
    aaa = 2;
  else
    % select only ErrP calibration data
    mk = mrk_selectEvents(mrk.classified, mrk.classified.pos>Cnt.T(1));
    mk_misc = mrk_selectEvents(mrk.misc, mrk.misc.pos>Cnt.T(1));

    % find machine and user errors
    mk_machine = mrk_selectClasses(mk_misc, 'machine_error');
    mk_machine = mrk_matchStimWithResp(mk, ...
                                       mk_machine, ...
                                      'missingresponse_policy', 'accept', ...
                                      'min_latency', -1500, ...
                                      'max_latency', 0);
    mk.machine_error = ~mk_machine.missingresponse;

    mk_user = mrk_selectClasses(mk_misc, 'user_error');
    if ~isempty(mk_user.toe)
      mk_user = mrk_matchStimWithResp(mk, ...
                                      mk_user, ...
                                     'missingresponse_policy', 'accept', ...
                                     'min_latency', -1500, ...
                                     'max_latency', 0);
      mk.user_error = ~mk_user.missingresponse;
    else
      mk.user_error = mk.error & ~mk.machine_error;
    end
    mk = mrk_addIndexedField(mk, {'machine_error', 'user_error'});

    mk = mrk_selectEvents(mk, find(~mk.user_error)); % exclude user errors
    mk.y = [mk.machine_error; ~mk.machine_error];
    mk.className = {'error', 'correct'};
    clear mk_misc mk_machine mk_user
  end
  
  %% artifact rejection (trials and/or channels)
  flds= {'reject_artifacts', 'reject_channels', ...
         'reject_artifacts_opts', 'clab'};
  if bbci_memo.data_reloaded || ...
        ~fieldsareequal(bbci_bet_memo_opt, opt, flds),
    clear anal
    anal.rej_trials= NaN;
    anal.rej_clab= NaN;
    if opt.reject_artifacts || opt.reject_channels,
      handlefigures('use', 'Artifact rejection', 1);
      set(gcf, 'Visible','off', ...
               'name',sprintf('%s: Artifact rejection', Cnt.short_title));
      [mk_clean , rClab, rTrials]= ...
          reject_varEventsAndChannels(Cnt, mk, [0 1000], ...
                                      'clab',clab, ...
                                      'do_multipass', 1, ...
                                      opt.reject_artifacts_opts{:}, ...
                                      'visualize', bbci.withgraphics);
      set(gcf,  'Visible','on');
      if opt.reject_artifacts,
        if not(isempty(rTrials)),
          %% TODO: make output class-wise
          fprintf('rejected: %d trial(s).\n', length(rTrials));
        end
        anal.rej_trials= rTrials;
        mk = mk_clean;
      end
      if opt.reject_channels,
        if not(isempty(rClab)),
          fprintf('rejected channels: <%s>.\n', vec2str(rClab));
        end
        anal.rej_clab= rClab;
      end
    end
  end
  if iscell(anal.rej_clab),   %% that means anal.rej_clab is not NaN
    clab(strpatternmatch(anal.rej_clab, clab))= [];
  end
  

  %% widely non_target
  if opt.widely_nontarget && aaa==1,
    % select subgroup of non-targets (save memory and avoid overlap)
    target= find(mk.y(1,:));
    widely_nontarget= find(mk.y(2,:) & ...
                           [mk.y(2,2:end), 1] & ...
                           [mk.y(2,3:end), 1, 1] & ...
                           [1, mk.y(2,1:end-1)] & ...
                           [1, 1, mk.y(2,1:end-2)] & ...
                           [1, 1, 1, mk.y(2,1:end-3)]);
    mk= mrk_chooseEvents(mk, union(target, widely_nontarget));
  end

  %% segmentation
  epo= cntToEpo(Cnt, mk, opt.disp_ival{aaa});
  epo= proc_selectChannels(epo, clab);
  epo= proc_baseline(epo, opt.ref_ival{aaa}, 'pos','beginning_exact');

  %% rejection of eyemovements based on max-min criterium 
  if opt.reject_eyemovements && opt.reject_eyemovements_crit.maxmin>0,
    epo_crit= proc_selectIval(epo, opt.reject_eyemovements_crit.ival);
    iArte= find_artifacts(epo_crit, opt.reject_eyemovements_crit.clab, ...
                          opt.reject_eyemovements_crit);
    fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
            length(iArte), opt.reject_eyemovements_crit.maxmin);
    clear epo_crit
    epo= proc_selectEpochs(epo, 'not',iArte);
    anal.eyemovement_trials= iArte;
  else
    anal.eyemovement_trials= NaN;
  end

  %% sgn r^2 values:
  epo_r= proc_r_square_signed(epo);
  if aaa==1
    epo_r.className= {'sgn r^2 ( T , NT )'};  %% just make it shorter
  else
    epo_r.className= {'sgn r^2 ( Err , NonErr )'};
  end

  %% find intervals:
  handlefigures('use','r^2 Matrix',1);
  if isempty(opt.cfy_ival{aaa}) || isequal(opt.cfy_ival{aaa}, 'auto'),
    set(gcf, opt_fig{:}, 'Visible','off', 'name',...
             sprintf('%s: r^2 Matrix', Cnt.short_title));
    [opt.cfy_ival{aaa}, nfo]= ...
        select_time_intervals(epo_r, 'visualize', 1, 'visu_scalps', 1, ...
                              'sort', 1, ...
                              'clab', clab, ...
                              'ival_pick_peak', opt.cfy_pick_peak{aaa});
    bbci_bet_message('[%g %g] ms\n', opt.cfy_ival{aaa}');
  else
    addpath([BCI_DIR 'investigation/teaching/ss09_analysis_of_neuronal_data']);
    clear nfo
    for ii= 1:size(opt.cfy_ival{aaa}, 1);
      nfo(ii).ival= opt.cfy_ival{aaa}(ii,:);
    end
    visualize_score_matrix(epo_r, nfo);
  end
  ival_scalps= visutil_correctIvalsForDisplay(opt.cfy_ival{aaa}, 'fs',epo.fs);
  set(gcf,  'Visible','on');
  anal.ival= opt.cfy_ival{aaa};
  pause(0.01)

  %% plot grid, erp, erp_r
  if bbci.withgraphics
    drawnow
    handlefigures('use','ERPs',1);
    set(gcf, opt_fig{:}, 'Visible','off', 'name',...
             sprintf('%s: ERP grid plot', Cnt.short_title));
    H= grid_plot(epo, mnt, defopt_erps, 'colorOrder',opt_scalp_erp.colorOrder);
    grid_addBars(epo_r, 'h_scale',H.scale);
    set(gcf,  'Visible','on');

    handlefigures('use','ERP Maps',1);
    set(gcf, opt_fig{:}, 'Visible','off', 'name',...
             sprintf('%s: ERP scalp maps', Cnt.short_title));
    H= scalpEvolutionPlusChannel(epo, mnt, opt.clab_erp, ival_scalps, ...
                                 opt_scalp_erp);
    grid_addBars(epo_r);
    set(gcf,  'Visible','on');

    if isempty(opt.clab_rsq) || isequal(opt.clab_rsq,'auto'),
      opt.clab_rsq= unique_unsort({nfo.peak_clab}, 4);
    end
    handlefigures('use','ERP r^2 Maps',1);
    set(gcf, opt_fig{:}, 'Visible','off', 'name',...
             sprintf('%s: ERP r^2 scalp maps', Cnt.short_title));
    scalpEvolutionPlusChannel(epo_r, mnt, opt.clab_rsq, ival_scalps, ...
                                opt_scalp_r);
    set(gcf,  'Visible','on');
    pause(0.01)
  end

  %% proc features
  epo= cntToEpo(Cnt, mk, opt.disp_ival{aaa});
  fv= proc_selectChannels(epo, opt.cfy_clab{aaa});
  fv= proc_baseline(fv, opt.ref_ival{aaa}, 'pos','beginning_exact');
  fv= proc_jumpingMeans(fv, opt.cfy_ival{aaa});

  %% do xvalidation:
  if opt.withclassification,
    opt_xv= strukt('sample_fcn',{'kfold' [1 10]}, ...
                   'loss','rocArea');
    [dum,dum,outTe] = xvalidation(fv, opt.model, opt_xv);
    me= val_confusionMatrix(fv, outTe, 'mode','normalized');
    remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
    bbci_bet_message(remainmessage,0);
  else
    remainmessage= '';
  end

  %% save results
  analyze{aaa}= strukt('features', fv, ...
                       'ival', opt.cfy_ival{aaa}, ...
                       'ref_ival', opt.ref_ival{aaa}, ...
                       'message', remainmessage, ...
                       'xValOutTe', outTe);
  analyze{aaa}= merge_structs(analyze{aaa}, anal);
  
  fprintf('\n\n*** Select intervals and save them to ''bbci.setup_opts.cfy_ival{%d}''.\n', aaa);
  fprintf('*** When finished type ''bbci.setup_opts.analyze_step = bbci.setup_opts.analyze_step + 1;''\n');
  fprintf('*** and re-run ''bbci_bet_analyze'' to continue with next step.\n');

end  

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select Nr. of Sequences:  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt.analyze_step == 3
  aaa = 1;
  if isfield(bbci, 'analyze')
    analyze = bbci.analyze;
  end
  % train on whole training set:
  fv = analyze{aaa}.features;
  [fv, nBlocks, nClasses]= proc_sortWrtStimulus(fv);  % events ordnen
  C = trainClassifier(fv, opt.model);
  out = applyClassifier(fv, opt.model, C);
  
  % evaluate output:
%   nShuffles = 1;
%   nDivisions = 5;
%   opt_xv= strukt('sample_fcn',{'kfold' [nShuffles nDivisions]});
  maxNrSequences = max(fv.block_idx);
  nTrials = max(fv.trial_idx);
  loss = zeros(maxNrSequences, 2);  
  for rr=1:maxNrSequences
    idx = fv.block_idx<=rr;
    
    % select only first rr outputs:
    outTe = out(idx);
    outTe = reshape(outTe, [nClasses, rr, nTrials]);
    [dum,x_hex]= min(squeeze(mean(outTe, 2))); % select the element with min avg classifier-output
    
    % select first rr labels:
    y = reshape(fv.y(1,idx), [nClasses, rr*nTrials]);
    y_hex = [1:6] * y(:,1:rr:end); % for each trial, take only one out of rr (because they are equal)
    
%     % Select relevant number of sequences:
%     fv = proc_selectEpochs(fv, fv.block_idx<=rr);
%     [fv, nBlocks, nClasses]= proc_sortWrtStimulus(fv);  % events ordnen
%     [dum,dum,outTe] = xvalidation(fv, opt.model, opt_xv);    
%     me= val_confusionMatrix(fv, outTe, 'mode','normalized');
%     remainmessage = sprintf('%d sequence: Correct Hits: %2.1f, Correct Miss: %2.1f\n',rr,100*me(1,1),100*me(2,2));
%     bbci_bet_message(remainmessage,0);
% 
%     % Compute accuracy
%     y = reshape(fv.y(1,:), [nClasses, rr*nTrials]);
%     y = repmat(y, [1 nShuffles]);
%     yy_hex = [1:6] * y(:,1:rr:end); % for each trial, take only one out of rr (because they are equal)
%     out = reshape(squeeze(outTe), [nClasses, rr, nTrials*nShuffles]);
%     [dum,ihex]= min(squeeze(mean(out, 2))); % select the element with min avg classifier-output
    
    % Compute accuracy:
    loss_trial= (x_hex~=y_hex);       % error for each level separately
    loss_symbol = loss_trial(1:2:end) | loss_trial(2:2:end);  % overall symbol selection error
    loss(rr,:)= [100*mean(loss_trial), 100*mean(loss_symbol)];
  end
  
  %% plot results
  opt.nr_sequences = find(100-loss(:,2) > opt.nr_sequences_threshold, 1);
  if isempty(opt.nr_sequences)
    opt.nr_sequences = maxNrSequences;
  end
  handlefigures('use','Nr. of Sequences',1); clf
  set(gcf, opt_fig{:}, 'Visible','off', 'name',...
           sprintf('%s: Nr. of Sequences', Cnt.short_title));
  plot(100-loss(:,1), '-ok');
  hold on
  plot(100-loss(:,2), '-ob');
  plot([opt.nr_sequences opt.nr_sequences], [0 100], ':k')
  text(opt.nr_sequences+0.1, 33, ['selected nr of sequences: ' num2str(opt.nr_sequences)])
  hold off
  ylabel('accuracy [%]');
  set(gca, 'XLim',[0 maxNrSequences+1], 'XTick',1:maxNrSequences, 'YLim',[0 105], 'YGrid','on');
  xlabel('repetitions [#]');
  legend({['trialwise: '  num2str(100-loss(opt.nr_sequences,1),3) '%'], ...
          ['symbolwise: ' num2str(100-loss(opt.nr_sequences,2),3) '%']}, 'Location','East')
  set(gcf,  'Visible','on');
  
  %% 
  fprintf('\n*** Choose nr of sequences and save them in ''bbci.setup_opts.nr_sequences''.\n');
  fprintf('*** When finished type ''bbci.setup_opts.analyze_step = bbci.setup_opts.analyze_step + 1;''\n');
  fprintf('*** and re-run ''bbci_bet_analyze'' to continue with next step.\n');
 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Choose Classifier Bias:      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt.analyze_step == 4
  aaa = 2;
  if isfield(bbci, 'analyze')
    analyze = bbci.analyze;
  end
  outTe = analyze{aaa}.xValOutTe;
  fv = analyze{aaa}.features;
  
  %% plot cl-output
  opt.nhist = 30;
  outmin = min(outTe); 	outmax = max(outTe);
  bias = linspace(outmin,outmax,opt.nhist);
  [dum idx] = min(abs(bias-opt.ErrP_bias));
  bias(idx) = opt.ErrP_bias;
  pos = outTe(1,logical(fv.y(2,:)),:); % corrects
  neg = outTe(1,logical(fv.y(1,:)),:); % errors
  hpos = histc(pos,bias) / numel(pos);
  hneg = histc(neg,bias) / numel(neg);
  handlefigures('use','Cl-Bias',1); clf
  set(gcf, opt_fig{:}, 'Visible','off', 'name',...
           sprintf('%s: Classifier Output', Cnt.short_title));
  subplot(211)
  plot(bias, hneg);
  hold all
  plot(bias, hpos);
  plot([opt.ErrP_bias opt.ErrP_bias], [0 max([hpos hneg])], ':k')
  hold off
  xlabel('classifier output')
  legend({'Error' 'Correct'})
  title(['selected bias: ' num2str(opt.ErrP_bias, 3)], 'FontSize', 13)
  
  %% investigate bias:
  label = [-1 1]*fv.y;
  accuracy = zeros(size(bias));
  correct_hits = zeros(size(bias));
  false_alarms = zeros(size(bias));
  for bb=1:length(bias)
    out = sign(outTe - bias(bb));
    corE = (out==-1 & label==-1); % correct classified labels
    corN = (out==1 & label==1); % correct classified labels
    ch = out(label==-1)==-1; % true positives
    fa = out(label==1)==-1; % false positives
    accuracy(bb) = 100*(sum(corE)/sum(label==-1) + sum(corN)/sum(label==1))/2;
    correct_hits(bb) = 100*mean(ch);
    false_alarms(bb) = 100*mean(fa);
  end
  
  subplot(212)
  plot(bias, accuracy, ':dk')
  hold on
  plot(bias, correct_hits, ':^b')
  plot(bias, false_alarms, ':vr')
  plot([opt.ErrP_bias opt.ErrP_bias], [0 100], ':k')
  hold off
  idx = find(bias==opt.ErrP_bias);
  legend(['normalized accuracy: ' num2str(accuracy(idx),3) '%'], ...
         ['correct hits: ' num2str(correct_hits(idx),3) '%'], ...
         ['false alarms: ' num2str(false_alarms(idx),3) '%'])
  xlabel('bias')
  ylabel('[%]')
  set(gcf,  'Visible','on');
  
  %% plot ROC
  handlefigures('use','ROC',1);
  set(gcf, opt_fig{:}, 'Visible','off', 'name',...
           sprintf('%s: ROC', Cnt.short_title));
  [roc, auc] = val_rocCurve(fv, outTe, ...
                            'plot', 1, ...
                            'ignoreNaN', 1);
	hold on
  plot(false_alarms(idx)/100, correct_hits(idx)/100, '.r', 'MarkerSize', 40)
  hold off
  set(gcf,  'Visible','on'); pause(0.05)
  
  fprintf('\n*** Select bias and save it to ''bbci.setup_opts.ErrP_bias''.\n');
  fprintf('*** When finished run ''bbci_bet_finish'' to save settings.\n');
  
end

bbci_memo.data_reloaded= 0;
