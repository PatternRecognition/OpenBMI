%BBCI_BET_ANALYZE_CSPP_AUTO
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
%   analyze  struct, will be passed on to bbci_bet_finish_lapcsp
%   bbci : updated
%   bbci_memo : updated

% claudia.sannelli@tu-berlin.de, May-2011


% Everything that should be carried over to bbci_bet_finish_csp must go
% into a variable of this name:
analyze= [];

default_grd= ...
  sprintf('scale,FC3,FCz,FC4,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
default_colDef= {'left', 'right',   'foot',  'rest'; ...
  [0.8 0 0.8], [0 0.7 0], [0 0 1], [0 0 0]};
area_LH= {'FFC5-3','FC5-3','CFC5-3','C5-3','CCP5-3','CP5-3','PCP5-3'};
area_C= {'FFC1-2','FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2','PCP1-2','P1-2'};
%area_C= {'FFC1-2','FCz','CFC1-2','Cz','CCP1-2','CPz','PCP1-2','Pz'};
area_RH= {'FFC4-6','FC4-6','CFC4-6','C4-6','CCP4-6','CP4-6','PCP4-6'};
default_motorarea= {area_LH, area_C, area_RH};
patch_list= {'small','sixnew', 'six','large','eightnew','eight','eightsparse','ten','eleven','eleven_to_anterior','twelve','eighteen','twentytwo'};
patch_centers= {'FC3,1,z,2,4','CFC3,1,z,2,4', 'C3,1,z,2,4', 'CCP3,1,z,2,4','CP3,1,z,2,4', 'Pz'};

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
  'clab_csp', '*', ...
  'patch_list', patch_list, ...
  'patch_centers', patch_centers, ...
  'kFolds', 4, ...
  'loss', 'rocAreaMore', ...
  'require_complete_neighborhood', 0, ...
  'selband_opt', [], ...
  'selival_opt', [], ...
  'usedPat', 'auto', ...
  'usedPat_csp', 'auto', ...
  'do_laplace', 1, ...
  'visu_laplace', 1, ...
  'visu_band', [5 35], ...
  'visu_ival', [-500 5000], ...
  'visu_ref_ival', [-500 0], ...
  'visu_classes', '*', ...
  'visu_specmaps', 1, ...
  'visu_erdmaps', 1, ...
  'visu_features', 1, ...
  'grd', default_grd, ...
  'colDef', default_colDef, ...
  'motorarea', default_motorarea, ...
  'verbose', 1);
%% TODO: optional visu_specmaps, visu_erdmaps

bbci_bet_memo_opt= ...
  set_defaults(bbci_bet_memo_opt, ...
  'nPat', NaN, ...
  'nPat_csp', NaN, ...
  'usedPat', NaN, ...
  'usedPat_csp', NaN, ...
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
if bbci_bet_memo_opt.nPat_csp~=opt.nPat_csp ...
    & ~strcmpi(opt.usedPat_csp, 'auto') ...
    & isequal(bbci_bet_memo_opt.usedPat_csp, opt.usedPat_csp),
  opt.usedPat_csp= 1:2*opt.nPat_csp;
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
opt.patch_centers= clab(chanind(clab, opt.patch_centers));
nCenters= length(opt.patch_centers);

if opt.reject_outliers,
  %% TODO: execute only if neccessary
  if opt.verbose,
    bbci_bet_message('checking for outliers\n');
  end
  fig1= handlefigures('use', 'trial-outlierness', 1);
  fig2= handlefigures('use', 'channel-outlierness', 1);
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
memo_opt.patch= opt.patch;
memo_opt.patch_centers= opt.patch_centers;
clear analyze mean_loss std_loss
for ci= 1:size(class_combination,1),

  disp(class_combination)
  classes= this_mrk.className(class_combination(ci,:));
  if strcmp(classes{1},'right') | strcmp(classes{2},'left'),
    class_combination(ci,:)= fliplr(class_combination(ci,:));
    classes= this_mrk.className(class_combination(ci,:));
  end
  if isfield(opt, 'events'),
    disp(['select events ' int2str(opt.events{ci}(1)) ':' int2str(opt.events{ci}(end))])
    trials_idx= 1:length(mrk.pos);
    trials_idx(rTrials) = [];
    [trials_ci trials_ci_idx]= intersect(trials_idx, opt.events{ci});     
    mrk2= mrk_selectEvents(this_mrk, trials_ci_idx);
  else
    mrk2= this_mrk;
  end
  if size(class_combination,1)>1,
    fprintf('\ninvestigating class combination <%s> vs <%s>\n', classes{:});
  end
  mrk2= mrk_selectClasses(mrk2, classes);  
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
  if opt.ival(2) < 2750
    opt.ival= [opt.ival(1) 2750];
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

  fv= cntToEpo(cnt_flt, mrk2, opt.ival, 'clab', clab);
  
  opt.patch= memo_opt.patch;
  if isequal(opt.patch, 'auto') | isempty(opt.patch) | ~isfield(opt, 'patch')
    %% Selection of best patch
    if opt.verbose>=2,
      bbci_bet_message('selecting patch and centers\n');
    end
    proc.memo= {'W','linDerClab'};
    proc.apply= ['fv = proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
      'fv = proc_variance(fv); ' ...
      'fv = proc_logarithm(fv);'];
    opt_xv= strukt('sample_fcn',{'chronKfold', opt.kFolds},'std_of_means',0,'loss',opt.loss(1:end-4));
    idx1= [];
    idx2= [];
    for ipatch= 1:length(opt.patch_list)      
      opt.patch_centers= memo_opt.patch_centers;
      nCenters= length(opt.patch_centers);
      disp([int2str(ipatch) ', ' int2str(nCenters)])      
      [requClabs Wall neighborClabs]= getClabForLaplacian(clab, 'clab', opt.patch_centers, 'filter_type', opt.patch_list{ipatch}, 'require_complete_neighborhood', opt.require_complete_neighborhood);
      centers_to_rm= [];
      for ic= 1:nCenters
        if isempty(neighborClabs{ic})
          centers_to_rm(end+1)= ic;
        end
      end
      opt.patch_centers(centers_to_rm)= [];
      proc.sneakin= {'patch_centers', opt.patch_centers, 'clab_csp', opt.clab_csp};
      proc.train= ['[fv_csp, wtmp]= proc_csp_auto(proc_selectChannels(fv, clab_csp), ''patterns'', ' int2str(opt.nPat_csp) '); ' ...
        '[fv_patch,w_patch]= proc_cspp_auto(fv, ''patterns'', ' int2str(opt.nPat) ', ''require_complete_neighborhood'', ' ...
        int2str(opt.require_complete_neighborhood) ', ''patch'', ''' opt.patch_list{ipatch} ''', ''patch_centers'', patch_centers); ' ...
        'w_csp = zeros(length(fv.clab),size(wtmp,2)); ' 'w_csp(chanind(fv,fv_csp.origClab),:) = wtmp; ' ...
        'W= cat(2, w_patch, w_csp); ' 'linDerClab= cat(2,fv_patch.clab,fv_csp.clab); ' ...
        'fv= proc_linearDerivation(fv, W, ''clab'', linDerClab); ' 'fv= proc_variance(fv); ' 'fv= proc_logarithm(fv);'];
      [loss1(ipatch), dummy, out_test_xv]= xvalidation(fv, opt.model, opt_xv, 'proc', proc);
      if findstr(opt.loss,'0_1')
        loss2(ipatch)= loss_rocArea(fv.y, out_test_xv);
        loss1(ipatch)= 100*(loss1(ipatch));
      else
        loss2(ipatch)= 100*mean(loss_0_1(fv.y, out_test_xv));
      end
    end
    [minlossxv bestpatchxv]= min(loss1);
    idx1= find(loss1==minlossxv);
    if length(idx1) > 1
      [minlossxv2 bestpatchxv]= min(loss2(idx1));
      idx2= find(loss2(idx1)==minlossxv2);
      if findstr(opt.loss,'Less') > 1
        idx2= idx2(1);
      else
        idx2= idx2(end);
      end
      idx1= idx1(idx2);
      bestpatchxv= idx1;
    end
    opt.patch= patch_list{bestpatchxv};
    %% TODO: check how to give this info properly in output
    if opt.verbose>=2,
      bbci_bet_message('%s','Chosen PATCH: \n', opt.patch);
    end
    disp(opt.patch);
  end

  opt.patch_centers= memo_opt.patch_centers;
  [requClabs dummy2 neighborClabs]= getClabForLaplacian(clab, 'clab', opt.patch_centers, 'filter_type', opt.patch, 'require_complete_neighborhood', opt.require_complete_neighborhood);
  patch_clab= cell(1,nCenters);
  centers_to_rm= [];
  for ic= 1:nCenters
    if ~isempty(neighborClabs{ic})
      patch_clab{ic}= cat(2, opt.patch_centers{ic}, neighborClabs{ic});
    else
      centers_to_rm= [centers_to_rm ic];
    end
  end
  opt.patch_centers(centers_to_rm)= [];
  nCenters= length(opt.patch_centers);
  patch_clab(centers_to_rm)= [];
  
  anal.patch= opt.patch;
  anal.patch_centers= opt.patch_centers;
  
  %% TODO: when simulations are finished change the order of proc_cspp_auto outputs
  if opt.verbose>=2,
    bbci_bet_message('calculating PATCHES\n');
  end
  if ischar(opt.usedPat) & strcmpi(opt.usedPat,'auto'),
    [fv_patch, w_patch_memo, score, A, usedPatMatrix]= proc_cspp_auto(fv, 'patterns', opt.nPat, ...
      'patch_centers', opt.patch_centers, 'patch', opt.patch, ...
      'require_complete_neighborhood', opt.require_complete_neighborhood, ...
      'patch_clab', patch_clab);
  else
    [fv_patch, w_patch_memo, score, A, usedPatMatrix]= proc_cspp_auto(fv, 'patterns', opt.nPat, ...
      'patch_centers', opt.patch_centers, 'patch', opt.patch, ...
      'require_complete_neighborhood', opt.require_complete_neighborhood, ...
      'patch_selectPolicy', 'equalperclass', ...
      'patch_clab', patch_clab);
  end
  if opt.verbose>=2,
    bbci_bet_message('calculating CSP\n');
  end
  if ischar(opt.usedPat_csp) & strcmpi(opt.usedPat_csp,'auto'),
    [fv_csp, w_csp_memo_tmp, score_csp, A_csp, usedPat_csp]= proc_csp_auto(proc_selectChannels(fv, opt.clab_csp), 'patterns', opt.nPat_csp);
  else
    [fv_csp, w_csp_memo_tmp, score_csp, A_csp]= proc_csp3(proc_selectChannels(fv, opt.clab_csp), 'patterns', opt.nPat_csp);
  end
  w_csp_memo= zeros(length(fv.clab), size(w_csp_memo_tmp,2));
  w_csp_memo(chanind(fv, fv_csp.origClab),:) = w_csp_memo_tmp;

  if bbci.withgraphics && bbci.withclassification,
    if opt.verbose>=2,
      bbci_bet_message('Creating Figure CSP\n');
    end
    handlefigures('use','CSP');
    %     pause(2);
    set(gcf, fig_opt{:},  ...
      'name',sprintf('%s: CSP <%s> vs <%s>', Cnt.short_title, classes{:}));
    opt_scalp_csp= strukt('colormap', cmap_greenwhitelila(31));
    if ischar(opt.usedPat_csp) & strcmpi(opt.usedPat_csp,'auto'),
      plotCSPanalysis(proc_selectChannels(fv, opt.clab_csp), mnt_adaptMontage(mnt, opt.clab_csp), w_csp_memo_tmp, A_csp, score_csp, opt_scalp_csp, ...
        'row_layout',1, 'title','');
    else
      plotCSPanalysis(proc_selectChannels(fv, opt.clab_csp), mnt_adaptMontage(mnt, opt.clab_csp), w_csp_memo_tmp, A_csp, score_csp, opt_scalp_csp, ...
        'mark_patterns', opt.usedPat_csp);
    end
    handlefigures('next_fig','CSP')
    drawnow;
  end

  fv_patch= proc_variance(fv_patch);
  fv_patch= proc_logarithm(fv_patch);
  if ~ischar(opt.usedPat) | ~strcmpi(opt.usedPat, 'auto'),
    fv_patch= proc_selectChannels(fv_patch, opt.usedPat);
    w_patch_memo= w_patch_memo(:,opt.usedPat);
    usedPatMatrix= usedPatMatrix(opt.usedPat,:,:);
  end

  fv_csp= proc_variance(fv_csp);
  fv_csp= proc_logarithm(fv_csp);
  if ~ischar(opt.usedPat_csp) | ~strcmpi(opt.usedPat_csp,'auto'),
    fv_csp= proc_selectChannels(fv_csp, opt.usedPat_csp);
    w_csp_memo= w_csp_memo(:,opt.usedPat_csp);
  end

  spat_w= cat(2, w_patch_memo, w_csp_memo);
  linDerClab= cat(2,fv_patch.clab, fv_csp.clab);
  features= proc_linearDerivation(fv, spat_w, 'clab', linDerClab);
  features= proc_variance(features);
  features= proc_logarithm(features);

  %% BB: this validation takes the selection of patches and csp filters
  %% which have been selected from the complete data set!
  if bbci.withclassification,
    opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
      'std_of_means',0, ...
      'verbosity',0, ...
      'progress_bar',0);
    [loss,loss_std]= xvalidation(features, opt.model, opt_xv);
    bbci_bet_message('CSPP global: %4.1f +/- %3.1f\n',100*loss,100*loss_std);
    remainmessage= sprintf('CSPP global: %4.1f +/- %3.1f', 100*loss,100*loss_std);

    proc.memo= {'W','linDerClab'};
    if ischar(opt.usedPat) & strcmpi(opt.usedPat,'auto') & ischar(opt.usedPat_csp) & strcmpi(opt.usedPat_csp,'auto')
      proc.sneakin= {'patch_centers', opt.patch_centers, 'clab_csp', opt.clab_csp, 'patch_clab', patch_clab};
      proc.train= ['[fv_csp, wtmp]= proc_csp_auto(proc_selectChannels(fv, clab_csp), ''patterns'', ' int2str(opt.nPat_csp) '); ' ...
        '[fv_patch,w_patch]= proc_cspp_auto(fv, ''patterns'', ' int2str(opt.nPat) ', ''patch_clab'', patch_clab, ''require_complete_neighborhood'', ' ...
        int2str(opt.require_complete_neighborhood) ', ''patch'', ''' opt.patch ''', ''patch_centers'', patch_centers); ' ...
        'w_csp = zeros(length(fv.clab),size(wtmp,2)); ' 'w_csp(chanind(fv,fv_csp.origClab),:) = wtmp; ' ...
        'W= cat(2, w_patch, w_csp); ' ...
        'linDerClab= cat(2,fv_patch.clab,fv_csp.clab); ' ...
        'fv= proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
        'fv= proc_variance(fv); ' ...
        'fv= proc_logarithm(fv);'];
      proc.apply= ['fv = proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
        'fv = proc_variance(fv); ' ...
        'fv = proc_logarithm(fv);'];
      [loss,loss_std]= xvalidation(fv, opt.model, opt_xv, 'proc', proc);
      bbci_bet_message('CSPP auto inside: %4.1f +/- %3.1f\n', ...
        100*loss,100*loss_std);
      remainmessage= sprintf('%s\nCSPP auto inside: %4.1f +/- %3.1f', ...
        remainmessage, 100*loss,100*loss_std);
    else
      proc.sneakin= {'clab_csp', opt.clab_csp, 'patch_centers', opt.patch_centers, 'patch_clab', patch_clab};
      proc.train= ['[fv_csp, wtmp]= proc_csp3(proc_selectChannels(fv, clab_csp), ''patterns'', ' int2str(opt.nPat_csp) '); ' ...
        '[fv_patch, w_patch]= proc_cspp_auto(fv, ''patterns'', ' int2str(opt.nPat) ', ''patch_clab'', patch_clab, ''require_complete_neighborhood'', ' ...
        int2str(opt.require_complete_neighborhood) ', ''patch'', ''' opt.patch ''', ''patch_centers'', patch_centers, ''patch_selectPolicy'', ''equalperclass''); ' ...
        'w_csp = zeros(length(fv.clab),size(wtmp,2)); ' 'w_csp(chanind(fv,fv_csp.origClab),:) = wtmp; ' ...
        'W= cat(2, w_patch, w_csp); ' ...
        'linDerClab= cat(2,fv_patch.clab,fv_csp.clab); ' ...
        'fv= proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
        'fv= proc_variance(fv); ' ...
        'fv= proc_logarithm(fv);'];
      proc.apply= ['fv = proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
        'fv = proc_variance(fv); ' ...
        'fv = proc_logarithm(fv);'];
      [loss,loss_std]= xvalidation(fv, opt.model, opt_xv, 'proc', proc);
      bbci_bet_message('CSPP inside: %4.1f +/- %3.1f\n', ...
        100*loss,100*loss_std);
      remainmessage = sprintf('%s\nCSPP inside: %4.1f +/- %3.1f', ...
        remainmessage, 100*loss,100*loss_std);

      if ~isequal(opt.usedPat_csp, 1:2*opt.nPat_csp) |  ~isequal(opt.usedPat, 1:2*opt.nPat)

        if ischar(opt.usedPat) & strcmpi(opt.usedPat,'auto')
          proc_train_patch= ['[fv_patch, w_patch]= proc_cspp_auto(fv, ''patterns'', ' int2str(opt.nPat) ', ''require_complete_neighborhood'', ' ...
            int2str(opt.require_complete_neighborhood) ', ''patch'', ''' opt.patch ''', ''patch_clab'', patch_clab, ''patch_centers'', patch_centers); '];
          patch_message= 'patch auto';
        elseif ~isequal(opt.usedPat, 1:2*opt.nPat),
          usedPat= opt.usedPat;
          proc.sneakin= cat(2, proc.sneakin, {'w_patch_memo', w_patch_memo});
          proc_train_patch= ['[fv_patch, w_patch]= proc_cspp_auto(fv, ''patterns'', w_patch_memo, ''require_complete_neighborhood'', ' ...
            int2str(opt.require_complete_neighborhood) ', ''patch'', ''' ...
            opt.patch ''', ''patch_selectPolicy'', ''matchfilters'', ''patch_centers'', patch_centers, ''patch_clab'', patch_clab); '];
          patch_message= 'selPatch';
        else
          usedPat= opt.usedPat;
          proc.sneakin= cat(2, proc.sneakin, {'patch_centers', opt.patch_centers});
          proc_train_patch= ['[fv_patch, w_patch]= proc_cspp_auto(fv, ''patterns'', ' int2str(opt.nPat) ', ''patch_clab'', patch_clab, ''require_complete_neighborhood'', ' ...
            int2str(opt.require_complete_neighborhood) ', ''patch'', ''' opt.patch ''', ''patch_centers'', patch_centers, ''patch_selectPolicy'', ''equalperclass''); '];
          patch_message= [int2str(opt.nPat*2) ' patches'];
        end

        if ischar(opt.usedPat_csp) & strcmpi(opt.usedPat_csp,'auto')
          proc_train_csp= ['[fv_csp, wtmp]= proc_csp_auto(proc_selectChannels(fv, clab_csp), ''patterns'', ' int2str(opt.nPat_csp) '); '];
          csp_message= 'csp auto';
        elseif ~isequal(opt.usedPat_csp, 1:2*opt.nPat_csp),
          usedPat_csp= opt.usedPat_csp;
          proc.sneakin= cat(2, proc.sneakin, {'w_csp_memo', w_csp_memo});
          proc_train_csp= '[fv_csp, wtmp]= proc_csp_auto(fv, ''patterns'', w_csp_memo, ''selectPolicy'', ''matchfilters''); ';
          csp_message= 'selCSP';
        else
          usedPat_csp= opt.usedPat_csp;
          proc_train_csp= ['[fv_csp, wtmp]= proc_csp_auto(proc_selectChannels(fv, clab_csp), ''patterns'', ' int2str(opt.nPat_csp) '); '];
          csp_message= [int2str(opt.nPat_csp) 'CSPs'];
        end

        proc.train= [proc_train_csp proc_train_patch ...
          'w_csp = zeros(length(fv.clab),size(wtmp,2)); ' ...
          'w_csp(chanind(fv,fv_csp.origClab),:) = wtmp; ' ...
          'W= cat(2, w_patch, w_csp); ' ...
          'linDerClab= cat(2,fv_patch.clab,fv_csp.clab); ' ...
          'fv= proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
          'fv= proc_variance(fv); ' ...
          'fv= proc_logarithm(fv);'];
        proc.apply= ['fv = proc_linearDerivation(fv, W, ''clab'', linDerClab); ' ...
          'fv = proc_variance(fv); ' ...
          'fv = proc_logarithm(fv);'];

        [loss,loss_std]= xvalidation(fv, opt.model, opt_xv, 'proc', proc);
        bbci_bet_message('%s, %s: %4.1f +/- %3.1f\n', ...
          csp_message, patch_message, 100*loss,100*loss_std);
        remainmessage= sprintf('%s\n%s, %s: %4.1f +/- %3.1f', ...
          remainmessage, csp_message, patch_message, 100*loss,100*loss_std);

      end
    end
    mean_loss(ci)= loss;
    std_loss(ci)= loss_std;
  end
  % leave fv to be used in case of adaptation

  analyze(ci)= ...
    merge_structs(anal, strukt('clab', clab, ...
    'filt_a', filt_a, ...
    'filt_b', filt_b, ...
    'spat_w', spat_w, ...
    'spat_w_csp', w_csp_memo, ...
    'spat_w_patch', w_patch_memo, ...
    'patch', opt.patch, ...
    'patch_centers', opt.patch_centers, ...
    'patch_clab', patch_clab, ...
    'cspp_clab', fv_patch.clab, ...
    'csp_clab', fv_csp.clab, ...
    'usedPat', usedPatMatrix(:,1)', ...
    'usedPat_csp', usedPat_csp, ...
    'usedPatMatrix', usedPatMatrix, ...
    'features', features, ...
    'message', remainmessage));

  clear fv_csp fv_patch

  %% Visualization of spectra and ERD/ERS curves
  if bbci.withgraphics,
    %     close all
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
    pause(2);
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
      nClasses= length(mrk2.className);
      %% calc spectrum without laplace for scalp maps
      spec= cntToEpo(Cnt, mrk2, opt.ival);
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
      %       pause(2);
      nCols= nClasses*2 + 1;
      for cc= 1:nClasses,
        ax_bp(cc)= subplot(2, nCols, cc*2+[-1 0]);
      end
      for cc= 1:length(spec_r.className),
        ax_bpr(cc)= subplot(2, nCols, nCols+cc*2+[0 1]);
      end
      H= scalpPatterns(spec, mnt, opt.band, opt_scalp, ...
        'subplot',ax_bp, ...
        'mark_channels', opt.patch_centers(usedPatMatrix(:,2)));
      H= scalpPatterns(spec_r, mnt, opt.band, opt_scalp_r, ...
        'subplot',ax_bpr, ...
        'mark_channels', opt.patch_centers(usedPatMatrix(:,2)), ...
        'mark_patterns', mp);
      %                     'scalePos','WestOutside');
      if nClasses==3,
        set(H.text(setdiff(1:3,mp)), 'FontWeight','normal');
      end
      handlefigures('next_fig','Spectra Maps');
      drawnow;
      clear spec spec_r
    end

    if opt.verbose>=2,
      bbci_bet_message('Creating figure(s) for ERD\n');
    end
    handlefigures('use','ERD',size(opt.band,1));
    pause(2);
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
    erd= cntToEpo(erd, mrk2, opt.visu_ival);
    erd= proc_baseline(erd, opt.visu_ref_ival, 'classwise',1);
    erd_rsq= proc_r_square_signed(proc_selectClasses(erd, classes));

    h= grid_plot(erd, mnt, opt_grid);
    grid_markIval(opt.ival);
    grid_addBars(erd_rsq, ...
      'h_scale',h.scale, ...
      'box', 'on', ...
      'colormap', cmap_posneg(31), ...
      'cLim', 'sym');
    clear erd erd_rsq;
    handlefigures('next_fig','ERD');
    drawnow;

    if opt.visu_erdmaps,
      %% calc ERD curves without laplace for ERD maps
      nClasses= length(mrk2.className);
      erd= proc_channelwise(cnt_flt, 'envelope', 'ms_msec', 200);
      clear cnt_flt
      if isempty(opt.visu_ref_ival),
        ival= opt.ival;
      else
        ival= [opt.visu_ref_ival(1) opt.ival(2)];
      end
      erd= cntToEpo(erd, mrk2, ival);
      erd= proc_baseline(erd, opt.visu_ref_ival, 'classwise',1);
      erd_r= proc_r_square_signed(erd);
      if nClasses==3,
        erd_r= proc_selectClasses(erd_r, [1 3 2]);  %% assumes nClasses=3
      end
      mp= intersect(strpatternmatch(['*' mrk2.className{1} '*'], ...
        erd_r.className), ...
        strpatternmatch(['*' mrk2.className{2} '*'], ...
        erd_r.className));

      handlefigures('use','ERD Maps',1);
      %       pause(2);
      nCols= nClasses*2 + 1;
      for cc= 1:nClasses,
        ax_erd(cc)= subplot(2, nCols, cc*2+[-1 0]);
      end
      for cc= 1:length(erd_r.className),
        ax_erdr(cc)= subplot(2, nCols, nCols+cc*2+[0 1]);
      end
      H= scalpPatterns(erd, mnt, opt.ival, opt_scalp_erp, ...
        'subplot',ax_erd, ...
        'mark_channels',  opt.patch_centers(usedPatMatrix(:,2)));
      H= scalpPatterns(erd_r, mnt, opt.ival, opt_scalp_r, ...
        'subplot',ax_erdr, ...
        'mark_channels',  opt.patch_centers(usedPatMatrix(:,2)), ...
        'mark_patterns', mp);
      %                     'scalePos','WestOutside');
      if nClasses==3,
        set(H.text(setdiff(1:3,mp)), 'FontWeight','normal');
      end
      handlefigures('next_fig','ERD Maps');
      drawnow;
      clear erd erd_r
    end
    if opt.visu_features
      bbci_addplot_csppFeatures;
    end
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

if opt.verbose>=2,
  bbci_bet_message('Finished analysis\n');
end
