%BBCI_BET_ANALYZE_ADAPTATIONSTUDY_SEASON1
%
%Description.
% Analyzes data provided by bbci_bet_prepare according to the
% parameters specified in the struct 'bbci'.
% It provides features for bbci_bet_finish_cspauto
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
%  ival: interval on which CSP is performed, 'auto' means automatic selection.
%  band: frequency band on which CSP is performed, 'auto' means
%     automatic selection.
%  nPat: number of CSP patterns which are considered from each side of the
%     eigenvalue spectrum. Note that not neccessarily all of these are
%     used for classification, see opt.usedPat.
%  usedPat: vector specifying the indices of the CSP filters that should
%     be used for classification, 'auto' means automatic selection.
%  do_laplace: do Laplace spatial filtering for automatic selection
%     of ival/band. If opt.do_laplace is set to 0, the default value of
%     will also be set to opt.visu_laplace 0 (but can be overwritten).
%  visu_laplace: do Laplace filtering for grid plots of Spectra/ERDs.
%     If visu_laplace is a string, it is passed to proc_laplace. This
%     can be used to use alternative geometries, like 'vertical'.
%  visu_band: Frequency range for which spectrum is shown.
%  visu_ival: Time interval for which ERD/ERS curves are shown.
%  visu_classes: Classes for which Spectra and ERD curves are drawn,
%     default '*'.
%  grd: grid to be used in the grid plots of spectra and ERDs.
%
%Output:
%   analyze  struct, will be passed on to bbci_bet_finish_csp
%   bbci : updated
%   bbci_memo : updated

% blanker@cs.tu-berlin.de, Oct-2008


% Everything that should be carried over to bbci_bet_finish_csp must go
% into a variable of this name:
analyze = [];

default_grd= ...
    sprintf('scale,FC3,FCz,FC4,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
default_colDef= {'left', 'right',   'foot',  'rest'; ...
                 [0.8 0 0.8], [0 0.7 0], [0 0 1], [0 0 0]};
area_LH= {'FFC5-3','FC5-3','CFC5-3','C5-3','CCP5-3','CP5-3','PCP5-3'};
area_C= {'FFC1-2','FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2','PCP1-2','P1-2'};
%area_C= {'FFC1-2','FCz','CFC1-2','Cz','CCP1-2','CPz','PCP1-2','Pz'};
area_RH= {'FFC4-6','FC4-6','CFC4-6','C4-6','CCP4-6','CP4-6','PCP-6'};
default_motorarea= {area_LH, area_C, area_RH};

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'reject_artifacts', 1, ...
                 'reject_channels', 1, ...
                 'reject_opts', {}, ...
                 'reject_outliers', 0, ...
                 'check_ival', [500 4500], ...
                 'ival', 'auto', ...
                 'default_ival', [1000 3500], ...
                 'repeat_bandselection', 1, ...
                 'selband_opt', [], ...
                 'selival_opt', [], ...
                 'usedPat', 'auto', ...
                 'do_laplace', 1, ...
                 'visu_laplace', 1, ...
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
                 'verbose', 1);
%% TODO: optional visu_specmaps, visu_erdmaps

bbci_bet_memo_opt= ...
    set_defaults(bbci_bet_memo_opt, ...
                 'nPat', NaN, ...
                 'usedPat', NaN, ...
                 'band', NaN);

if isdefault.default_ival & opt.verbose,
  msg= sprintf('default ival not defined in bbci.setup_opts, using [%d %d]', ...
               opt.default_ival);
  warning(msg);
end
if isdefault.visu_laplace & ~opt.do_laplace,
  opt.visu_laplace= 0;
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
%% when nPat was changed, but usedPat was not, define usedPat
if bbci_bet_memo_opt.nPat~=opt.nPat ...
      & ~strcmpi(opt.usedPat, 'auto') ...
      & isequal(bbci_bet_memo_opt.usedPat, opt.usedPat),
  opt.usedPat= 1:2*opt.nPat;
end


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
      set(gcf, fig_opt{:},  ...
               'name',sprintf('%s: Artifact rejection', Cnt.short_title));
    end
    [mk_clean , rClab, rTrials]= ...
        reject_varEventsAndChannels(Cnt, mrk, opt.check_ival, ...
                                    'clab',clab, ...
                                    'do_multipass', 1, ...
                                    opt.reject_opts{:}, ...
                                    'visualize', bbci.withgraphics);
    if bbci.withgraphics,
      handlefigures('next_fig','Artifact rejection');
      drawnow;
    end
    if opt.reject_artifacts,
      if length(rTrials)>0 | opt.verbose,
        %% TODO: make output class-wise
        fprintf('rejected: %d trial(s).\n', length(rTrials));
      end
      anal.rej_trials= rTrials;
    end
    if opt.reject_channels,
      if length(rClab)>0 | opt.verbose,
        fprintf('rejected channels: <%s>.\n', vec2str(rClab));
      end
      anal.rej_clab= rClab;
    end
  end
end
if iscell(anal.rej_clab),   %% that means anal.rej_clab is not NaN
  clab= setdiff(clab, anal.rej_clab);
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


%% Automatic selection of parameters (band, ival)
opt.band= memo_opt.band;    %% for classes='auto' do sel. for each combination
opt.ival= memo_opt.ival;
band_fresh_selected= 0;
if isequal(opt.band, 'auto') | isempty(opt.band);
  bbci_bet_message('No band specified, select an optimal one: ');
  if ~isequal(opt.ival, 'auto') & ~isempty(opt.ival),
    ival_for_bandsel= opt.ival;
  else
    ival_for_bandsel= opt.default_ival;
    if ~opt.repeat_bandselection,
      bbci_bet_message('\nYou should run bbci_bet_analyze a 2nd time.\n pre-selection of band: ');
    end
  end
  opt.band= select_bandnarrow(Cnt, mrk2, ival_for_bandsel, ...
                              opt.selband_opt, 'do_laplace',opt.do_laplace);
  bbci_bet_message('[%g %g] Hz\n', opt.band);
  band_fresh_selected= 1;
end

%% This reusing may fail for bbci.classes='auto', so we have commented it out
%if ~isequal(opt.band, bbci_bet_memo_opt.band) | ...
%      ~isequal(opt.filtOrder, bbci_bet_memo_opt.filtOrder) | ...
%      ~exist('cnt_flt','var'),
  [filt_b,filt_a]= butter(opt.filtOrder, opt.band/Cnt.fs*2);
  clear cnt_flt
  cnt_flt= proc_filt(Cnt, filt_b, filt_a);
%  if opt.verbose>=2,
%    bbci_bet_message('Data filtered\n');  
%  end
%elseif opt.verbose>=2,
%  bbci_bet_message('Filtered data reused\n');    
%end

if isequal(opt.ival, 'auto') | isempty(opt.ival),
  bbci_bet_message('No ival specified, automatic selection: ');
  opt.ival= select_timeival(cnt_flt, mrk2, ...
                            opt.selival_opt, 'do_laplace',opt.do_laplace);
  bbci_bet_message('[%i %i] msec.\n', opt.ival);
end

if opt.repeat_bandselection & band_fresh_selected & ...
      ~isequal(opt.ival, ival_for_bandsel),
  bbci_bet_message('Redoing selection of frequency band for new interval: ');
  first_selection= opt.band;
  opt.band= select_bandnarrow(Cnt, mrk2, opt.ival, ...
                              opt.selband_opt, 'do_laplace',opt.do_laplace);
  bbci_bet_message('[%g %g] Hz\n', opt.band);
  if ~isequal(opt.band, first_selection),
    clear cnt_flt
    [filt_b,filt_a]= butter(opt.filtOrder, opt.band/Cnt.fs*2);
    cnt_flt= proc_filt(Cnt, filt_b, filt_a);
  end
end
anal.band= opt.band;
anal.ival= opt.ival;

fv= cntToEpo(cnt_flt, mrk2, opt.ival, 'clab',clab);

%% Selection of discriminative Laplace channels
if opt.verbose>=2,
  bbci_bet_message('selecting LAP channels\n');
end
[fv_lap, lap_w]= proc_laplacian(fv);
fv_lap= proc_variance(fv_lap);
fv_lap= proc_logarithm(fv_lap);
f_score= proc_rfisherScore(fv_lap);
score= zeros(length(opt.motorarea)*opt.nlaps_per_area, 1);
sel_idx= score;
for ii= 1:length(opt.motorarea),
  aidx= chanind(f_score, opt.motorarea{ii});
  [dmy, si]= sort(abs(f_score.x(aidx)), 2, 'descend');
  idx= (ii-1)*opt.nlaps_per_area + [1:opt.nlaps_per_area];
  score(idx)= f_score.x(aidx(si(1:opt.nlaps_per_area)));
  sel_idx(idx)= aidx(si(1:opt.nlaps_per_area));
end
sel_clab= strhead(f_score.clab(sel_idx));
if opt.verbose,
  fprintf('selected Laplacian channels: ');
  for ii= 1:length(sel_clab),
    fprintf('%s (%.2f)  ', sel_clab{ii}, score(ii));
  end
  fprintf('\n');
end

%% Visualization of spectra and ERD/ERS curves

if bbci.withgraphics,
  disp_clab= getClabOfGrid(mnt);
  if opt.visu_laplace,
    requ_clab= getClabForLaplacian(Cnt, disp_clab);
  else
    requ_clab= disp_clab;
  end
  if opt.verbose>=2,
    bbci_bet_message('Creating figure for spectra\n');
  end
  if diff(opt.ival)>=1000,
    spec= cntToEpo(Cnt, this_mrk, opt.ival, 'clab',requ_clab);
  else
    bbci_bet_message('Enlarging interval to calculate spectra\n');
    spec= cntToEpo(Cnt, this_mrk, opt.ival(1)+[0 1000], 'clab',requ_clab);
  end
  handlefigures('use','Spectra',1);
  set(gcf, fig_opt{:}, 'name',...
           sprintf('%s: spectra in [%d %d] ms', Cnt.short_title, opt.ival));
  if opt.visu_laplace,
    if ischar(opt.visu_laplace),
      spec= proc_laplacian(spec, opt.visu_laplace);
    else
      spec= proc_laplacian(spec);
    end
  end
  spec= proc_spectrum(spec, opt.visu_band, kaiser(Cnt.fs,2));
  spec_rsq= proc_r_square_signed(proc_selectClasses(spec,classes));
    
  h= grid_plot(spec, mnt, opt_grid_spec);
  grid_markIval(opt.band);
  grid_addBars(spec_rsq, ...
               'h_scale', h.scale, ...
               'box', 'on', ...
               'colormap', cmap_posneg(31), ...
               'cLim', 'sym');
    
  clear spec_rsq spec
  handlefigures('next_fig','Spectra');
  drawnow;

  if opt.visu_specmaps,
    nClasses= length(this_mrk.className);
    %% calc spectrum without laplace for scalp maps
    spec= cntToEpo(Cnt, this_mrk, opt.ival);
    [Tt,Nn,NTnt]=size(spec.x);
    LWin=Cnt.fs;
    if Tt<Cnt.fs
      LWin=Tt;
    end;
    spec= proc_spectrum(spec, opt.band, kaiser(LWin,2));
    spec_r= proc_r_square_signed(spec);
    if nClasses==3,
      spec_r= proc_selectClasses(spec_r, [1 3 2]);
    end
    mp= intersect(strpatternmatch(['*' mrk2.className{1} '*'], ...
                                  spec_r.className), ...
                  strpatternmatch(['*' mrk2.className{2} '*'], ...
                                  spec_r.className));
    handlefigures('use','Spectra Maps',1);
    nCols= nClasses*2 + 1;
    for cc= 1:nClasses,
      ax_bp(cc)= subplot(2, nCols, cc*2+[-1 0]);
    end
    for cc= 1:length(spec_r.className),
      ax_bpr(cc)= subplot(2, nCols, nCols+cc*2+[0 1]);
    end
    H= scalpPatterns(spec, mnt, opt.band, opt_scalp, ...
                     'subplot',ax_bp, ...
                     'mark_channels', sel_clab);
    H= scalpPatterns(spec_r, mnt, opt.band, opt_scalp_r, ...
                     'subplot',ax_bpr, ...
                     'mark_channels', sel_clab, ...
                     'mark_patterns', mp);
%                     'scalePos','WestOutside');
    if nClasses==3,
      set(H.text(setdiff(1:3,mp)), 'FontWeight','normal');
    end
    handlefigures('next_fig','Spectra Maps');
    clear spec spec_r
  end
  
  if opt.verbose>=2,
    bbci_bet_message('Creating figure(s) for ERD\n');
  end
  handlefigures('use','ERD',size(opt.band,1));
  set(gcf, fig_opt{:},  ...
           'name',sprintf('%s: ERD for [%g %g] Hz', ...
                          Cnt.short_title, opt.band));
  erd= proc_selectChannels(cnt_flt, requ_clab);
  if opt.visu_laplace,
    if ischar(opt.visu_laplace),
      erd= proc_laplacian(erd, opt.visu_laplace);
    else
      erd= proc_laplacian(erd);
    end
  end
  erd= proc_envelope(erd, 'ms_msec', 200);
  erd= cntToEpo(erd, this_mrk, opt.visu_ival);
  erd= proc_baseline(erd, opt.visu_ref_ival, 'classwise',1);
  erd_rsq= proc_r_square_signed(proc_selectClasses(erd, classes));
  
  h = grid_plot(erd, mnt, opt_grid);
  grid_markIval(opt.ival);
  grid_addBars(erd_rsq, ...
               'h_scale',h.scale, ...
               'box', 'on', ...
               'colormap', cmap_posneg(31), ...
               'cLim', 'sym');
  drawnow;
  clear erd erd_rsq;
  handlefigures('next_fig','ERD');

  if opt.visu_erdmaps,
    %% calc ERD curves without laplace for ERD maps
    nClasses= length(this_mrk.className);
    erd= proc_channelwise(cnt_flt, 'envelope', 'ms_msec', 200);
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
    
    handlefigures('use','ERD Maps',1);
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
                     'mark_channels', sel_clab, ...
                     'mark_patterns', mp);
%                     'scalePos','WestOutside');
    if nClasses==3,
      set(H.text(setdiff(1:3,mp)), 'FontWeight','normal');
    end
    handlefigures('next_fig','ERD Maps');
    clear erd erd_r
  end
  
end
clear cnt_flt

if opt.verbose>=2,
  bbci_bet_message('calculating CSP\n');
end

if ischar(opt.usedPat) & strcmpi(opt.usedPat,'auto'),
  [fv2, csp_w, la, A]= proc_csp_auto(fv, 'patterns',opt.nPat);
else
  [fv2, csp_w, la, A]= proc_csp3(fv, 'patterns',opt.nPat);
end

if bbci.withgraphics | bbci.withclassification,
  if opt.verbose>=2,
    bbci_bet_message('Creating Figure CSP\n');
  end
  handlefigures('use','CSP');
  set(gcf, fig_opt{:},  ...
      'name',sprintf('%s: CSP <%s> vs <%s>', Cnt.short_title, classes{:}));
  opt_scalp_csp= strukt('colormap', cmap_greenwhitelila(31));
  if ischar(opt.usedPat) & strcmpi(opt.usedPat,'auto'),
    plotCSPanalysis(fv, mnt, csp_w, A, la, opt_scalp_csp, ...
                    'row_layout',1, 'title','');
  else
    plotCSPanalysis(fv, mnt, csp_w, A, la, opt_scalp_csp, ...
                    'mark_patterns', opt.usedPat);
  end
  drawnow;
end

fv_csp= proc_variance(fv2);
fv_csp= proc_logarithm(fv_csp);
if ~ischar(opt.usedPat) | ~strcmpi(opt.usedPat,'auto'),
  fv_csp.x= fv_csp.x(:,opt.usedPat,:);
  csp_w= csp_w(:,opt.usedPat);
end
global_w= csp_w;

spat_w= [lap_w, csp_w];
features= proc_appendChannels(fv_lap, fv_csp);
features= proc_flaten(features);

%% BB: this validation takes the selection of 3 laplacian filters
%% which have been selected from the complete data set!
if bbci.withclassification,
  opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
                 'std_of_means',0, ...
                 'verbosity',0, ...
                 'progress_bar',0);
  [loss,loss_std] = xvalidation(features, opt.model, opt_xv);
  bbci_bet_message('CSP/LAP global: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
  remainmessage= sprintf('CSP/LAP global: %4.1f +/- %3.1f', ...
                          100*loss,100*loss_std);
  if ischar(opt.usedPat) & strcmpi(opt.usedPat,'auto'),
    proc= strukt('memo', {'csp_w', 'lap_w'});
    proc.sneakin= {'sel_clab',sel_clab};
    proc.train= ['[fv_lap,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
                 '[fv,csp_w]= proc_csp_auto(fv, ' int2str(opt.nPat) '); ' ...
                 'fv= proc_appendChannels(fv_lap, fv); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv); '];
    proc.apply= ['fv= proc_linearDerivation(fv, [lap_w, csp_w]); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv); '];
    [loss,loss_std] = xvalidation(fv, opt.model, opt_xv, 'proc',proc);
    bbci_bet_message('CSP/LAP auto inside: %4.1f +/- %3.1f\n', ...
                     100*loss,100*loss_std);
    remainmessage = sprintf('%s\nCSP/LAP auto inside: %4.1f +/- %3.1f', ...
                            remainmessage, 100*loss,100*loss_std);
  else
    proc= strukt('memo', {'csp_w','lap_w'});
    proc.sneakin= {'sel_clab',sel_clab};
    proc.train= ['[fv_lap,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
                 '[fv,csp_w]= proc_csp3(fv, ' int2str(opt.nPat) '); ' ...
                 'fv= proc_appendChannels(fv_lap, fv); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv);'];
    proc.apply= ['fv= proc_linearDerivation(fv, [lap_w, csp_w]); ' ...
                 'fv= proc_variance(fv); ' ...
                 'fv= proc_logarithm(fv);'];
    [loss,loss_std] = xvalidation(fv, opt.model, opt_xv, 'proc',proc);
    bbci_bet_message('CSP/LAP inside: %4.1f +/- %3.1f\n', ...
                     100*loss,100*loss_std);
    remainmessage = sprintf('%s\nCSP/LAP inside: %4.1f +/- %3.1f', ...
                            remainmessage, 100*loss,100*loss_std);
    
    if ~isequal(opt.usedPat, 1:2*opt.nPat),
      proc.sneakin= {'global_w',global_w, 'sel_clab',sel_clab};
      proc.train=['[fv_lap,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
                  '[fv,csp_w]= proc_csp3(fv, ''patterns'',global_w, ' ...
                     '''selectPolicy'',''matchfilters''); ' ...
                  'fv= proc_appendChannels(fv_lap, fv); ' ...
                  'fv= proc_variance(fv); ' ...
                  'fv= proc_logarithm(fv);'];
      proc.apply=['fv= proc_linearDerivation(fv, [lap_w, csp_w]); ' ...
                  'fv= proc_variance(fv); ' ...
                  'fv= proc_logarithm(fv);'];
      [loss,loss_std]= xvalidation(fv, opt.model, opt_xv, 'proc',proc);
      bbci_bet_message('CSP/LAP selPat: %4.1f +/- %3.1f\n', ...
                       100*loss,100*loss_std);
      remainmessage = sprintf('%s\nCSP setPat: %4.1f +/- %3.1f', ...
                              remainmessage, 100*loss,100*loss_std);
    end
  end
  mean_loss(ci)= loss;
  std_loss(ci)= loss_std;
end
clear fv fv2

% Gather all information that should be saved in the classifier file
nComp= size(spat_w, 2);
isactive= ismember([1:nComp], ...
                   [chanind(fv_lap, sel_clab), size(lap_w,2)+1:nComp]);
analyze(ci)= ...
    merge_structs(anal, strukt('clab', clab, ...
                               'isactive', isactive, ...
                               'sel_lapclab', sel_clab, ...
                               'filt_a', filt_a, ...
                               'filt_b', filt_b, ...
                               'spat_w', spat_w, ...
                               'features', features, ...
                               'message', remainmessage));

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

if opt.verbose>=2,
  bbci_bet_message('Finished analysis\n');
end
