%BBCI_BET_ANALYZE_SELLAP
%
%Description.
% Analyzes data provided by bbci_bet_prepare according to the
% parameters specified in the struct 'bbci'.
% It provides features for bbci_bet_finish_sellap
% 
%Input (variables defined before calling this function):
% Cnt, mrk, mnt:  (loaded by bbci_bet_prepare)
% bbci:   struct, the general setup variable from bbci_bet_prepare
% bbci_memo: internal use
% opt    (copied by bbci_bet_analyze from bbci.setup_opts) a struct with fields
%  reject_artifacts:
%  reject_channels:
%  reject_opts: cell array of options which are passed to 
%     reject_varEventsAndChannels.
%  reject_outliers:
%  check_ival: interval which is checked for artifacts/outliers
%  ival: interval from which features are extracted, 'auto' means automatic selection.
%  band: frequency band from which band-power features are extracted, 'auto' means
%     automatic selection.
%  visu_band: Frequency range for which spectrum is shown.
%  visu_ival: Time interval for which ERD/ERS curves are shown.
%  visu_classes: Classes for which Spectra and ERD curves are drawn,
%     default '*'.
%  grd: grid to be used in the grid plots of spectra and ERDs.
%
%Output:
%   analyze  struct, will be passed on to bbci_bet_finish_sellap
%   bbci : updated
%   bbci_memo : updated

% blanker@cs.tu-berlin.de, Nov-2008

wstate= warning('off', 'bbci:multiple_channels');

% Everything that should be carried over to bbci_bet_finish_sellap must go
% into a variable of this name:
analyze = [];

default_grd= ...
    sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
default_colDef= {'left', 'right',   'foot',  'rest'; ...
                 [0.8 0 0.8], [0 0.7 0], [0 0 1], [0 0 0]};
area_LH= {'FFC5-3','FC5-3','CFC5-3','C5-3','CCP5-3','CP5-3','PCP5-3'};
area_C= {'FFC1-2','FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2','PCP1-2','P1-2'};
%area_C= {'FFC1-2','FCz','CFC1-2','Cz','CCP1-2','CPz','PCP1-2','Pz'};
area_RH= {'FFC4-6','FC4-6','CFC4-6','C4-6','CCP4-6','CP4-6','PCP4-6'};
default_motorarea= {area_LH, area_C, area_RH};

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'reject_artifacts', 1, ...
                 'reject_channels', 1, ...
                 'reject_opts', {}, ...
                 'reject_outliers', 0, ...
                 'check_ival', [500 4500], ...
                 'ival', 'auto', ...
                 'band', 'auto', ...
                 'default_ival', [1000 3500], ...
                 'selband_max', [6 35], ...
                 'selband_opt', [], ...
                 'selival_opt', [], ...
                 'visu_band', [5 35], ...
                 'visu_ival', [-500 5000], ...
                 'visu_ref_ival', [-500 0], ...
                 'visu_classes', '*', ...
                 'visu_specmaps', 1, ...
                 'visu_erdmaps', 0, ...
                 'grd', default_grd, ...
                 'colDef', default_colDef, ...
                 'motorarea', default_motorarea, ...
                 'nlaps_per_area', 2, ...
                 'select_lap_for_each_band', 1, ...
                 'allow_reselecting_laps', 0, ...
                 'verbose', 1);

if isempty(opt.band),
  opt.band= 'auto';
end

bbci_bet_memo_opt= ...
    set_defaults(bbci_bet_memo_opt, ...
                 'band', NaN);

if isdefault.default_ival & opt.verbose,
  msg= sprintf('default ival not defined in bbci.setup_opts, using [%d %d]', ...
               opt.default_ival);
  warning(msg);
end



%% Prepare visualization

mnt= mnt_setGrid(mnt, opt.grd);
mnt= mnt_enlarge(mnt);
opt_grid= defopt_erps('scale_leftshift',0.075);
%% TODO: extract good channel (like 'CPz' here) from grid
opt_grid_spec= defopt_spec('scale_leftshift',0.075, ...
                           'xTickAxes','CPz');
clab_gr1= intersect(scalpChannels, getClabOfGrid(mnt));
if isempty(clab_gr1),
  clab_gr1= getClabOfGrid(mnt);
end
opt_grid.scaleGroup= {clab_gr1, {'EMG*'}, {'EOG*'}};
fig_opt= {'numberTitle','off', 'menuBar','none'};
if length(strpatternmatch(mrk.className, opt.colDef(1,:))) < ...
      length(mrk.className),
  if ~isdefault.colDef,
    warning('opt.colDef does not match with mrk.className');
  end
  nClasses= length(mrk.className);
  cols= mat2cell(cmap_rainbow(nClasses), ones(1,nClasses), 3)';
  opt.colDef= {mrk.className{:}; cols{:}};
end
opt_scalp= ...
    defopt_scalp_power('mark_properties',{'MarkerSize',7,'LineWidth',2});
opt_scalp_erp= ...
    defopt_scalp_erp('mark_properties',{'MarkerSize',7,'LineWidth',2});
opt_scalp_r= ...
    defopt_scalp_r('mark_properties',{'MarkerSize',7,'LineWidth',2});


%% Analysis starts here
clear fv*
clab= Cnt.clab(chanind(Cnt, opt.clab));

%% artifact rejection (trials and/or channels)
flds= {'reject_artifacts', 'reject_channels', ...
       'reject_opts', 'check_ival', 'clab'};
if bbci_memo.data_reloaded | ...
      ~fieldsareequal(bbci_bet_memo_opt, opt, flds),
  clear anal
  anal.rej_trials= NaN;
  anal.rej_clab= NaN;
  if opt.reject_artifacts | opt.reject_channels,
    if opt.verbose,
      fprintf('checking for artifacts and bad channels\n');
    end
    if bbci.withgraphics,
      handlefigures('use', 'Artifact rejection', 1);
      set(gcf, fig_opt{:},  'Visible','off', ...
               'name',sprintf('%s: Artifact rejection', Cnt.short_title));
    end
    [mk_clean , rClab, rTrials]= ...
        reject_varEventsAndChannels(Cnt, mrk, opt.check_ival, ...
                                    'clab',clab, ...
                                    'do_multipass', 1, ...
                                    opt.reject_opts{:}, ...
                                    'visualize', bbci.withgraphics);
    if bbci.withgraphics,
      set(gcf,  'Visible','on');
    end
    if opt.reject_artifacts,
      if not(isempty(rTrials)) | opt.verbose,
        %% TODO: make output class-wise
        fprintf('rejected: %d trial(s).\n', length(rTrials));
      end
      anal.rej_trials= rTrials;
    end
    if opt.reject_channels,
      if not(isempty(rClab)) | opt.verbose,
        fprintf('rejected channels: <%s>.\n', vec2str(rClab));
      end
      anal.rej_clab= rClab;
    end
  end
end
if iscell(anal.rej_clab),   %% that means anal.rej_clab is not NaN
  clab(strpatternmatch(anal.rej_clab, clab))= [];
end

if opt.reject_outliers,
%% TODO: execute only if neccessary
  if opt.verbose,
    bbci_bet_message('checking for outliers\n');
  end
  fig1 = handlefigures('use', 'trial-outlierness', 1);
  fig2 = handlefigures('use', 'channel-outlierness', 1);
  %% TODO: reject_outliers only on artifact free trials?
  %%  clarify relation of reject_articfacts and reject_outliers
  fv= cntToEpo(Cnt, mrk, opt.check_ival, 'clab',clab);
  [fv, anal.outl_trials]=  ...
      proc_outl_var(fv, ...
                    'display', bbci.withclassification,...
                    'handles', [fig1,fig2], ...
                    'trialthresh',bbci.setup_opts.threshold);
  %% TODO: output number of outlier trials (class-wise)
  clear fv
  handlefigures('next_fig', 'trial-outlierness');
  handlefigures('next_fig', 'channel-outlierness');
else
  anal.outl_trials= NaN;
end

kickout_trials= union(anal.rej_trials, anal.outl_trials);
kickout_trials(find(isnan(kickout_trials)))= [];
this_mrk= mrk_chooseEvents(mrk, setdiff(1:length(mrk.pos), kickout_trials));


if ~isequal(bbci.classes, 'auto'),
  class_combination= strpatternmatch(bbci.classes, this_mrk.className);
  if length(class_combination) < length(bbci.classes),
    error('not all specified classes found');
  end
  if opt.verbose,
    fprintf('using classes <%s> as specified\n', vec2str(bbci.classes));
  end
else
  class_combination= nchoosek(1:size(this_mrk.y,1), 2);
end


%% Specific investigation of binary class combination(s) start
memo_opt.band= opt.band;
memo_opt.ival= opt.ival;
clear analyze mean_loss std_loss
for ci= 1:size(class_combination,1),
  
classes= this_mrk.className(class_combination(ci,:));
if strcmp(classes{1},'right') | strcmp(classes{2},'left'),
  class_combination(ci,:)= fliplr(class_combination(ci,:));
  classes= this_mrk.className(class_combination(ci,:));
end
if size(class_combination,1)>1,
  fprintf('\ninvestigating class combination <%s> vs <%s>\n', classes{:});
end
mrk2= mrk_selectClasses(this_mrk, classes);
opt_grid.colorOrder= choose_colors(this_mrk.className, opt.colDef);
opt_grid.lineStyleOrder= {'--','--','--'};
clidx= strpatternmatch(classes, this_mrk.className);
opt_grid.lineStyleOrder(clidx)= {'-'};
opt_grid_spec.lineStyleOrder= opt_grid.lineStyleOrder;
opt_grid_spec.colorOrder= opt_grid.colorOrder;

%% quick fix which needs a lot of memory:
%% TODO: ....
[cnt_lap, lap_w]= proc_laplacian(proc_selectChannels(Cnt,clab));

%% Automatic selection of parameters (band, ival)
opt.band= memo_opt.band;    %% for classes='auto' do sel. for each combination
opt.ival= memo_opt.ival;
band_fresh_selected= 0;
if isequal(opt.band, 'auto'),
  bbci_bet_message('No band specified, select an optimal one: ');
  if ~isequal(opt.ival, 'auto') & ~isempty(opt.ival),
    ival_for_bandsel= opt.ival;
  else
    ival_for_bandsel= opt.default_ival;
  end
  rng= [min(opt.selband_max(:)) max(opt.selband_max(:))];
  opt.band= select_bandnarrow(cnt_lap, mrk2, ival_for_bandsel, ...
                              opt.selband_opt, ...
                              'band',rng);
  bbci_bet_message('[%g %g] Hz\n', opt.band');
  band_fresh_selected= 1;
  nBands= size(opt.selband_max,1);
else
  nBands= size(opt.band,1);
end

%% This reusing may fail for bbci.classes='auto', so we have commented it out
%if ~isequal(opt.band, bbci_bet_memo_opt.band) | ...
%      ~isequal(opt.filtOrder, bbci_bet_memo_opt.filtOrder) | ...
%      ~exist('cnt_flt','var'),
  [filt_b,filt_a]= butters(opt.filtOrder, opt.band/Cnt.fs*2);
  clear cnt_flt
  cnt_flt= proc_filterbank(cnt_lap, filt_b, filt_a);
%  if opt.verbose>=2,
%    bbci_bet_message('Data filtered\n');  
%  end
%elseif opt.verbose>=2,
%  bbci_bet_message('Filtered data reused\n');    
%end

if isequal(opt.ival, 'auto') | isempty(opt.ival),
  bbci_bet_message('No ival specified, automatic selection: ');
  opt.ival= select_timeival(cnt_flt, mrk2, ...
                            opt.selival_opt);
  bbci_bet_message('[%i %i] msec.\n', opt.ival);
end

if band_fresh_selected & ...
      (~isequal(opt.ival, ival_for_bandsel) | nBands>1),
  bbci_bet_message('Redoing selection of frequency band for new interval: ');
  first_selection= opt.band;
  if nBands==1,
    opt.band= select_bandnarrow(cnt_lap, mrk2, opt.ival, ...
                                opt.selband_opt);
  else
    opt.band= zeros(nBands, 2);
    for ib= 1:nBands,
      opt.band(ib,:)= select_bandnarrow(cnt_lap, mrk2, opt.ival, ...
                                        opt.selband_opt, ...
                                        'band',opt.selband_max(ib,:));
    end
  end
  bbci_bet_message('  [%g %g] Hz\n', opt.band');
  if ~isequal(opt.band, first_selection),
    clear cnt_flt
    [filt_b,filt_a]= butters(opt.filtOrder, opt.band/Cnt.fs*2);
    cnt_flt= proc_filterbank(cnt_lap, filt_b, filt_a);
  end
end
anal.band= opt.band;
anal.ival= opt.ival;

%This would select the desired channels, but mess their ordering
% (due to presence of multiple channels, e.g. 'C3 lap_flt1' and 'C3 lap_flt2')
% -> mismatch of classifier with online processed data
%fv_lap= cntToEpo(cnt_flt, mrk2, opt.ival, 'clab',clab);
clab_flt= strpatternmatch(strcat(clab,'*'), cnt_flt.clab);
fv_lap= cntToEpo(cnt_flt, mrk2, opt.ival, 'clab',clab_flt);

%% Selection of discriminative Laplace channels
if opt.verbose>=2,
  bbci_bet_message('selecting LAP channels\n');
end
fv_lap= proc_variance(fv_lap);
fv_lap= proc_logarithm(fv_lap);
f_score= proc_rfisherScore(fv_lap);
tmpBands= nBands;
nC= length(f_score.clab)/nBands;
if nBands>1 & ~opt.select_lap_for_each_band,
  xx= reshape(f_score.x, [nC nBands]);
  [mm, mi]= max(abs(xx), [], 2);
  vmi= [1:nC]' + [(mi-1)*nC];
  f_score.x= f_score.x(vmi);
  f_score.clab= f_score.clab(1:nC);
%  f_score.clab= strcat(strhead(f_score.clab(1:nC)), ' lap');
  tmpBands= 1;
end
score= zeros([length(opt.motorarea) opt.nlaps_per_area tmpBands]);
sel_idx= score;
for ii= 1:length(opt.motorarea),
  for jj= 1:opt.nlaps_per_area,
    for kk= 1:tmpBands,
      aidx= chanind(f_score, opt.motorarea{ii});
      bidx= (kk-1)*nC + [1:nC];
      %bidx= strpatternmatch(['*' int2str(kk)], f_score.clab);
      idx= intersect(aidx, bidx);
      [dmy, mi]= max(abs(f_score.x(idx)));
      score(ii,jj,kk)= f_score.x(idx(mi));
      sel_idx(ii,jj,kk)= idx(mi);
      %% this prevents that the same channels is selected
      %% for different frequency bands:
      %cidx= chanind(f_score, strhead(f_score.clab(idx(mi))));
      %f_score.x(cidx)= 0;  %% avoid selecting this channel again
      f_score.x(idx(mi))= 0;  %% avoid selecting this channel again
    end
  end
end
sel_cflab= f_score.clab(sel_idx(:)');  %% label of (channel + filter)
sel_clab= strhead(f_score.clab);
sel_clab= sel_clab(sel_idx);           %% label of channel
if opt.verbose,
  fprintf('selected Laplacian channels: ');
  if tmpBands>1,
    for kk= 1:tmpBands,
      fprintf('\n  [%g %g]: ',anal.band(kk,:));
      for ii= 1:length(opt.motorarea),
        for jj= 1:opt.nlaps_per_area,
          fprintf('%s (%.2f)  ', sel_clab{ii,jj,kk}, score(ii,jj,kk));
        end
      end
    end
  else
    for ii= 1:numel(sel_clab),
      fprintf('%s (%.2f)  ', sel_clab{ii}, score(ii));
    end
  end
  fprintf('\n');
end
sel_clab= unique(sel_clab)';


%% Visualization of spectra and ERD/ERS curves

if bbci.withgraphics,
  disp_clab= getClabOfGrid(mnt);
  requ_clab= getClabForLaplacian(Cnt, disp_clab);
  if opt.verbose>=2,
    bbci_bet_message('Creating figure for spectra\n');
  end
  if diff(opt.ival)>=1000,
    winlen= Cnt.fs;
    spec_ival= opt.ival;
  else
    winlen= Cnt.fs/2;
    if diff(opt.ival)<500,
      bbci_bet_message('Enlarging interval to calculate spectra\n');
      spec_ival=  mean(opt.ival) + [-250 250];
    else
      spec_ival= opt.ival;
    end
  end
  spec= cntToEpo(cnt_lap, this_mrk, spec_ival, 'clab',requ_clab);
  handlefigures('use','Spectra',1);
  set(gcf, fig_opt{:}, 'Visible','off', 'name',...
           sprintf('%s: spectra in [%d %d] ms', Cnt.short_title, opt.ival));
  spec= proc_spectrum(spec, opt.visu_band, kaiser(winlen,2));
  spec_rsq= proc_r_square_signed(proc_selectClasses(spec,classes));
    
  h= grid_plot(spec, mnt, opt_grid_spec);
  grid_markIval(opt.band);
  grid_addBars(spec_rsq, 'h_scale', h.scale, 'cLim', 'sym');
    
  clear spec_rsq spec
  set(gcf,  'Visible','on');

  if opt.visu_specmaps,
    nClasses= length(this_mrk.className);
    %% calc spectrum without laplace for scalp maps
    spec= cntToEpo(Cnt, this_mrk, spec_ival);
    rng= [min(opt.band(:)) max(opt.band(:))];
    spec= proc_spectrum(spec, rng, kaiser(winlen,2));
    spec_r= proc_r_square_signed(spec);
    if nClasses==3,
      spec_r= proc_selectClasses(spec_r, [1 3 2]);
    end
    mp= intersect(strpatternmatch(['*' mrk2.className{1} '*'], ...
                                  spec_r.className), ...
                  strpatternmatch(['*' mrk2.className{2} '*'], ...
                                  spec_r.className));
    nCols= nClasses*2 + 1;
    handlefigures('use','Spectra Maps',nBands);
    for i= 1:nBands,
      set(gcf, fig_opt{:},  'Visible','off', ...
               'name',sprintf('%s: Band Power Maps for [%g %g] Hz', ...
                              Cnt.short_title, opt.band(i,:)));
      for cc= 1:nClasses,
        ax_bp(cc)= subplot(2, nCols, cc*2+[-1 0]);
      end
      for cc= 1:length(spec_r.className),
        ax_bpr(cc)= subplot(2, nCols, nCols+cc*2+[0 1]);
      end
      H= scalpPatterns(spec, mnt, opt.band(i,:), opt_scalp, ...
                       'subplot',ax_bp, ...
                       'mark_channels', sel_clab);
      H= scalpPatterns(spec_r, mnt, opt.band(i,:), opt_scalp_r, ...
                       'subplot',ax_bpr, ...
                       'newcolormap', 1, ...
                       'mark_channels', sel_clab, ...
                       'mark_patterns', mp);
      %                     'scalePos','WestOutside');
      if nClasses==3,
        set(H.text(setdiff(1:3,mp)), 'FontWeight','normal');
      end
      handlefigures('next_fig','Spectra Maps');
    end
    clear spec spec_r
  end
  
  if opt.verbose>=2,
    bbci_bet_message('Creating figure(s) for ERD\n');
  end
  handlefigures('use','ERD',nBands);
  for i = 1:size(opt.band,1),
    set(gcf, fig_opt{:}, 'Visible','off', ...
             'name',sprintf('%s: ERD for [%g %g] Hz', ...
                            Cnt.short_title, opt.band(i,:)));
    erd= proc_selectChannels(cnt_flt, ...
                  strpatternmatch(['*flt' int2str(i)], cnt_flt.clab));
    erd= proc_selectChannels(erd, requ_clab);
    erd= proc_envelope(erd, 'ms_msec', 200);
    erd= cntToEpo(erd, this_mrk, opt.visu_ival);
    erd= proc_baseline(erd, opt.visu_ref_ival, 'classwise',1);
    erd.clab= strcat(strhead(erd.clab), ' lap');   %% get rid of '_flt'
    erd_rsq= proc_r_square_signed(proc_selectClasses(erd, classes));
    
    h = grid_plot(erd, mnt, opt_grid);
    grid_markIval(opt.ival);
    grid_addBars(erd_rsq, 'h_scale',h.scale, 'cLim', 'sym');
    handlefigures('next_fig','ERD');
  end
  clear erd erd_rsq;

  if opt.visu_erdmaps,
    %% calc ERD curves without laplace for ERD maps
    nClasses= length(this_mrk.className);
    erd= proc_selectChannels(cnt_flt, strpatternmatch(['*flt' int2str(i)], cnt_flt.clab));
    erd= proc_channelwise(erd, 'envelope', 'ms_msec', 200);
    if isempty(opt.visu_ref_ival),
      ival= opt.ival;
    else
      ival= [opt.visu_ref_ival(1) opt.ival(2)];
    end
    erd= cntToEpo(erd, this_mrk, ival);
    erd= proc_baseline(erd, opt.visu_ref_ival, 'classwise',1);
    erd_r= proc_r_square_signed(erd);
    if nClasses==3,
      erd_r= proc_selectClasses(erd_r, [1 3 2]);  %% assumes nClasses=3
    end
    mp= intersect(strpatternmatch(['*' mrk2.className{1} '*'], ...
                                  spec_r.className), ...
                  strpatternmatch(['*' mrk2.className{2} '*'], ...
                                  spec_r.className));
    
    handlefigures('use','ERD Maps',nBands);
    for i = 1:size(opt.band,1),
      set(gcf, fig_opt{:}, 'Visible','off', ...
               'name',sprintf('%s: ERD Maps for [%g %g] Hz', ...
                              Cnt.short_title, opt.band(i,:)));
      nCols= nClasses*2 + 1;
      for cc= 1:nClasses,
        ax_erd(cc)= subplot(2, nCols, cc*2+[-1 0]);
      end
      for cc= 1:length(erd_r.className),
        ax_erdr(cc)= subplot(2, nCols, nCols+cc*2+[0 1]);
      end
      H= scalpPatterns(erd, mnt, opt.ival, opt_scalp_erp, ...
                       'subplot',ax_erd, ...
                       'mark_channels', sel_clab);
      H= scalpPatterns(erd_r, mnt, opt.ival, opt_scalp_r, ...
                       'subplot',ax_erdr, ...
                       'newcolormap', 1, ...
                       'mark_channels', sel_clab, ...
                       'mark_patterns', mp);
      %                     'scalePos','WestOutside');
      if nClasses==3,
        set(H.text(setdiff(1:3,mp)), 'FontWeight','normal');
      end
      handlefigures('next_fig','ERD Maps');
    end
    clear erd erd_r
  end
  
end
clear cnt_flt

if bbci.withclassification,
  opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
                 'std_of_means',0, ...
                 'verbosity',0, ...
                 'progress_bar',0);
  fv= proc_selectChannels(fv_lap, strpatternmatch(sel_cflab, fv_lap.clab));
  [loss,loss_std] = xvalidation(fv, opt.model, opt_xv);
  bbci_bet_message('SELLAP global: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
  remainmessage= sprintf('SELLAP global: %4.1f +/- %3.1f', ...
                          100*loss,100*loss_std);

  if 0,  %% TODO: perform selection of LAPs inside cross-validation
  proc= strukt('memo', 'lap_w');
  proc.sneakin= {'sel_clab',sel_clab};
  proc.train= ['[fv,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
  proc.apply= ['fv= proc_linearDerivation(fv, lap_w); ' ...
               'fv= proc_variance(fv); ' ...
               'fv= proc_logarithm(fv);'];
  [loss,loss_std] = xvalidation(fv, opt.model, opt_xv, 'proc',proc);
  bbci_bet_message('SELLAP inside: %4.1f +/- %3.1f\n', ...
                   100*loss,100*loss_std);
  remainmessage = sprintf('%s\nSELLAP inside: %4.1f +/- %3.1f', ...
                          remainmessage, 100*loss,100*loss_std);
  end
    
  mean_loss(ci)= loss;
  std_loss(ci)= loss_std;
end
clear fv

% Gather all information that should be saved in the classifier file
analyze(ci)= ...
    merge_structs(anal, strukt('sel_lapclab', sel_clab, ...
                               'filt_a', filt_a, ...
                               'filt_b', filt_b, ...
                               'spat_w', lap_w, ...
                               'message', remainmessage));
if opt.allow_reselecting_laps,
  isactive= ismember([1:length(fv_lap.clab)], ...
                     strpatternmatch(sel_cflab, fv_lap.clab));
  analyze(ci).clab= clab;
  analyze(ci).isactive= isactive;
  analyze(ci).features= proc_flaten(fv_lap);
else
  [analyze(ci).clab, analyze(ci).spat_w]= getClabForLaplacian(Cnt, sel_clab);
  analyze(ci).features= proc_flaten(proc_selectChannels(fv_lap,sel_clab));
end

end  %% for ci 

if isequal(bbci.classes, 'auto'),
  [dmy, bi]= min(mean_loss + 0.1*std_loss);
  bbci.classes= this_mrk.className(class_combination(bi,:));
  bbci.class_selection_loss= [mean_loss; std_loss];
  analyze= analyze(bi);
  bbci_bet_message(sprintf('\nCombination <%s> vs <%s> chosen.\n', ...
                           bbci.classes{:}));
  if bi<size(class_combination,1),
    opt.ival= analyze.ival;    %% restore selection of best class combination
    opt.band= analyze.band;
    bbci_bet_message('Rerun bbci_bet_analyze again to see corresponding plots\n');
  end
end
bbci_memo.data_reloaded= 0;
warning(wstate.state, 'bbci:multiple_channels');

if opt.verbose>=2,
  bbci_bet_message('Finished analysis\n');
end
