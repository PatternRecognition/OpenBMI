function [bbci, data]= bbci_calibrate_csp_plus_lap(bbci, data)
%BBCI_CALIBRATE_CSP_PLUS_LAP - Calibrate for SMR Modulations with CSP plus some selected Laplacians
%
%This function is called by bbci_calibrate 
%(if BBCI.calibate.fcn is set to @bbci_calibrate_csp_plus_lap).
%Via BBCI.calibrate.settings, the details can be specified, see below.
%This calibration is meant to be used with bbci_adaptation_csp_plus_lap.
%
%Synopsis:
% [BBCI, DATA]= bbci_calibrate_csp_plus_lap(BBCI, DATA)
% 
%Arguments:
%  BBCI -  the field 'calibrate.settings' holds parameters specific to
%          calibrate CSP-based BCI processing.
%  DATA -  holds the calibration data
%  
%Output:
%  BBCI - Updated BBCI structure in which all necessary fields for
%     online operation are set, see bbci_apply_structures.
%  DATA - As input but added some information of the analysis that might
%     be reused in a second run
%
%BBCI.calibrate.settings may include the following parameters:
%  classes: [1x2 CELL of CHAR, or (default) 'auto'] Names of the two classes, 
%     which are to be discriminated. For classes = 'auto', all pairs of
%     available classes are investigated, and the one with best
%     xvalidation performance is chosen.
%  ival: [1x2 DOUBLE, or (default) 'auto'] interval on which CSP is
%     performed, 'auto' means automatic selection.
%  band: [1x2 DOUBLE, or (default) 'auto'] frequency band on which CSP is
%     performed, 'auto' means automatic selection.
%  clab: [CELL of CHAR] Labels of the channels that are used for 
%     visualization and calculating Laplacians.
%     Default is {'not','E*','Fp*','AF*','OI*','I*','*9','*10'}.
%  clab_csp: [CELL of CHAR] Labels of the channels that are used for 
%     CSP analysis.
%     Default is {'not','E*','Fp*','AF*','OI*','I*','*9','*10'}.
%  nPatters: [INT>0] number of CSP patterns which are considered from each
%     side of the eigenvalue spectrum. Note, that not neccessarily all of
%     these are used for classification, see settings.patterns.
%     Default is 3.
%  patterns: [1xN DOUBLE, or (deafult) 'auto'] vector specifying the
%     indices of the CSP filters that should be used for classification; 
%     'auto' means automatic selection.
%  area
%  naps_per_area
%  buffer
%  model: [CHAR or CELL] classification model.
%     Default is {'RLDAshrink', 'store_means',1, 'scaling',1}.
%  reject_artifacts:
%  reject_channels:
%  reject_artifacts_opts: cell array of options which are passed to 
%     reject_varEventsAndChannels.
%  check_ival: interval which is checked for artifacts
%  visu_band: Frequency range for which spectrum is shown.
%  visu_ival: Time interval for which ERD/ERS curves are shown.
%  grd: grid to be used in the grid plots of spectra and ERDs.
%
%Only the figures of the chosen class combination are visible, while the
%others are hidden. In order to make them visible, type
%>> set(cat(2, data.all_results.figure_handles), 'Visible','on')
%
%You might like to modify bbci.feature.ival after running this function.

% 01-2012 Benjamin Blankertz


opt= bbci.calibrate.settings;

default_clab=  {'not','E*','Fp*','AF*','OI*','I*','*9','*10'};
default_grd= ...
    sprintf(['scale,_,F3,Fz,F4,_,legend\n' ...
             'FC5,FC3,FC1,FCz,FC2,FC4,FC6\n' ...
             'C5,C3,C1,Cz,C2,C4,C6\n' ...
             'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
             'P5,P3,P1,Pz,P2,P4,P6']);
default_colDef= {'left',      'right',   'foot',  'rest'; ...
                 [0.8 0 0.8], [0 0.7 0], [0 0 1], [0 0 0]};
default_model= {'RLDAshrink', 'store_means',1, 'scaling',1};
area_LH= {'FFC5-3','FC5-3','CFC5-3','C5-3','CCP5-3','CP5-3','PCP5-3'};
area_C= {'FFC1-2','FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2','PCP1-2','P1-2'};
area_RH= {'FFC4-6','FC4-6','CFC4-6','C4-6','CCP4-6','CP4-6','PCP4-6'};
default_area= {area_LH, area_C, area_RH};

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'classes', 'auto', ...
                 'visu_ival', [-500 5000], ...
                 'visu_band', [5 35], ...
                 'clab', default_clab, ...
                 'clab_csp', default_clab, ...
                 'ival', 'auto', ...
                 'band', 'auto', ...
                 'nPatterns', 3, ...
                 'patterns', 'auto', ...
                 'area', default_area, ...
                 'nlaps_per_area', 2, ...
                 'buffer', 80, ...
                 'model', default_model, ...
                 'reject_artifacts', 0, ...
                 'reject_channels', 0, ... 
                 'reject_artifacts_opts', {'clab', default_clab}, ...
                 'reject_outliers', 0, ...
                 'check_ival', [500 4500], ...
                 'default_ival', [1000 3500], ...
                 'min_ival_length', 300, ...
                 'enlarge_ival_append', 'end', ...
                 'selband_opt', [], ...
                 'selival_opt', [], ...
                 'filtOrder', 5, ...
                 'grd', default_grd, ...
                 'colDef', default_colDef);


%% -- Prepare visualization --
%

mnt= mnt_setGrid(data.mnt, opt.grd);
opt_grid= defopt_erps('scale_leftshift',0.075);
%% TODO: extract good channel (like 'Pz' here) from grid
opt_grid_spec= defopt_spec('scale_leftshift',0.075, ...
                           'xTickAxes','Pz');

if sum(ismember(opt.colDef(1,:), data.mrk.className)) < ...
      length(data.mrk.className),
  if ~isdefault.colDef,
    warning('opt.colDef does not match with data.mrk.className');
  end
  nClasses= length(data.mrk.className);
  cols= mat2cell(cmap_rainbow(nClasses), ones(1,nClasses), 3)';
  opt.colDef= {data.mrk.className{:}; cols{:}};
end
opt_scalp= ...
    defopt_scalp_power2('mark_properties',{'MarkerSize',7,'LineWidth',2});
opt_scalp_erp= ...
    defopt_scalp_erp2('mark_properties',{'MarkerSize',7,'LineWidth',2});
opt_scalp_r= ...
    defopt_scalp_r2('mark_properties',{'MarkerSize',7,'LineWidth',2});


if ~data.isnew && isfield(data, 'result'),
  previous= data.result;
else
  previous= struct;
end

BC_result= [];
BC_result.mrk= data.mrk;
BC_result.clab= data.cnt.clab(chanind(data.cnt, opt.clab));

mrk_all= data.mrk;


%% --- Artifact rejection (trials and/or channels) based on variance criterion
%
flds= {'reject_artifacts', 'reject_channels', ...
       'reject_artifacts_opts', 'clab'};
if data.isnew || ~isfield(data, 'previous_settings') || ...
      ~fieldsareequal(opt, data.previous_settings, flds),
  BC_result.rejected_trials= NaN;
  BC_result.rejected_clab= NaN;
  if opt.reject_artifacts | opt.reject_channels,
    fig_set(3, 'name','Artifact rejection');
    [mk_clean , rClab, rTrials]= ...
        reject_varEventsAndChannels(data.cnt, mrk_all, opt.check_ival, ...
                                    'do_multipass', 1, ...
                                    'visualize', 1, ...
                                    opt.reject_artifacts_opts{:});
    if opt.reject_artifacts,
      bbci_log_write(data, 'Rejected: %d trial(s).', length(rTrials));
      BC_result.rejected_trials= rTrials;
    end
    if opt.reject_channels,
      bbci_log_write(data, 'Rejected channels: <%s>', vec2str(rClab));
      BC_result.rejected_clab= rClab;
    end
  else
    % Avoid confusion with old figure from previous run
    close_if_exists(3);
  end
  if iscell(BC_result.rejected_clab),   %% that means rejected_clab is not NaN
    cidx= find(ismember(BC_result.clab, BC_result.rejected_clab));
    BC_result.clab(cidx)= [];
  end
else
  result_flds= {'rejected_trials', 'rejected_clab', 'clab'};
  BC_result= copy_fields(BC_result, previous, result_flds);
end

if isequal(opt.classes, 'auto'),
  class_combination= nchoosek(1:size(mrk_all.y,1), 2);
else
  class_combination= find(ismember(mrk_all.className, opt.classes));
  if length(class_combination) < length(opt.classes),
    error('Not all specified classes were found.');
  end
  if length(class_combination) ~= 2,
    error('This calibration is only for binary classification.');
  end
end

memo_opt.band= opt.band;
memo_opt.ival= opt.ival;
clear mean_loss std_loss


%% -- Specific investigation of binary class combination(s) starts here --
%

for ci= 1:size(class_combination,1),
 
figno_offset= 4*(ci-1);
classes= mrk_all.className(class_combination(ci,:));
if strcmp(classes{1},'right') || strcmp(classes{2},'left'),
  class_combination(ci,:)= fliplr(class_combination(ci,:));
  classes= mrk_all.className(class_combination(ci,:));
end
msg= sprintf('\n** Class combination <%s> vs <%s> **\n', classes{:});
bbci_log_write(data, msg);
mrk2= mrk_selectClasses(mrk_all, classes);
BC_result.mrk= mrk2;
BC_result.classes= classes;

opt_grid.colorOrder= choose_colors(mrk_all.className, opt.colDef);
opt_grid.lineStyleOrder= {'--','--','--'};
clidx= find(ismember(mrk_all.className, classes));
opt_grid.lineStyleOrder(clidx)= {'-'};
opt_grid_spec.lineStyleOrder= opt_grid.lineStyleOrder;
opt_grid_spec.colorOrder= opt_grid.colorOrder;


%% --- Automatic selection of parameters (band, ival) ---
%

opt.band= memo_opt.band;    %% for classes='auto' do sel. for each combination
opt.ival= memo_opt.ival;
band_fresh_selected= 0;
if isequal(opt.band, 'auto') || isempty(opt.band);
  bbci_log_write(data, 'No band specified, automatic selection:');
  if ~isequal(opt.ival, 'auto') & ~isempty(opt.ival),
    ival_for_bandsel= opt.ival;
  else
    ival_for_bandsel= opt.default_ival;
  end
  opt.band= select_bandnarrow(data.cnt, mrk2, ival_for_bandsel, ...
                              opt.selband_opt);
  bbci_log_write(data, ' -> [%g %g] Hz', opt.band);
  band_fresh_selected= 1;
end

[filt_b,filt_a]= butter(opt.filtOrder, opt.band/data.cnt.fs*2);
clear cnt_flt
cnt_flt= proc_filt(data.cnt, filt_b, filt_a);

if isequal(opt.ival, 'auto') || isempty(opt.ival),
  bbci_log_write(data, 'No ival specified, automatic selection:');
  opt.ival= select_timeival(cnt_flt, mrk2, opt.selival_opt);
  bbci_log_write(data, ' -> [%i %i] msec', opt.ival);
end

if band_fresh_selected && ~isequal(opt.ival, ival_for_bandsel),
  bbci_log_write(data, 'Redoing selection of freq. band for new interval:');
  first_selection= opt.band;
  opt.band= select_bandnarrow(data.cnt, mrk2, opt.ival, opt.selband_opt);
  bbci_log_write(data, ' -> [%g %g] Hz', opt.band);
  if ~isequal(opt.band, first_selection),
    clear cnt_flt
    [filt_b,filt_a]= butter(opt.filtOrder, opt.band/data.cnt.fs*2);
    cnt_flt= proc_filt(data.cnt, filt_b, filt_a);
  end
end


%% -- Selection of discriminative Laplacians --
%

fv= cntToEpo(cnt_flt, mrk2, opt.ival, 'clab',BC_result.clab);
[fv_lap, lap_w]= proc_laplacian(fv);
fv_lap= proc_variance(fv_lap);
fv_lap= proc_logarithm(fv_lap);
f_score= proc_rfisherScore(fv_lap);
score= zeros(length(opt.area)*opt.nlaps_per_area, 1);
sel_idx= score;
for ii= 1:length(opt.area),
  aidx= chanind(f_score, opt.area{ii});
  [dmy, si]= sort(abs(f_score.x(aidx)), 2, 'descend');
  idx= (ii-1)*opt.nlaps_per_area + [1:opt.nlaps_per_area];
  score(idx)= f_score.x(aidx(si(1:opt.nlaps_per_area)));
  sel_idx(idx)= aidx(si(1:opt.nlaps_per_area));
end
sel_clab= strhead(f_score.clab(sel_idx));
msg= 'Selected Laplacian channels:';
for ii= 1:length(sel_clab),
  msg= sprintf('%s %s (%.2f) ', msg, sel_clab{ii}, score(ii));
end
bbci_log_write(data, msg);


%% -- Visualization of Spectra and ERD/ERS curves --
%

disp_clab= getClabOfGrid(mnt);
requ_clab= getClabForLaplace(data.cnt, disp_clab);

% -- Spectra --
name= sprintf('Spectra in [%d %d] ms', opt.ival);
fig_set(figno_offset+1, 'name',name, 'set', {'Visible','off'});
if diff(opt.ival)>=opt.min_ival_length,
  tmp_ival= opt.ival;
else
  bbci_log_write(data, '!Enlarging interval to calculate spectra.');
  switch(opt.enlarge_ival_append),
   case 'start',
    tmp_ival= opt.ival(2) + [-opt.min_ival_length 0];
   case 'end',
    tmp_ival= opt.ival(1) + [0 opt.min_ival_length];
   otherwise
    error('opt.enlarge_ival_append option unknown.')
  end
end
spec= cntToEpo(data.cnt, mrk_all, tmp_ival, 'clab',requ_clab);

spec= proc_laplacian(spec);
if data.cnt.fs>size(spec.x,1)
  winlen= size(spec.x,1);
else
  winlen= data.cnt.fs;
end
spec= proc_spectrum(spec, opt.visu_band, kaiser(winlen, 2));
spec_rsq= proc_r_square_signed(proc_selectClasses(spec, classes));

h= grid_plot(spec, mnt, opt_grid_spec);
grid_markIval(opt.band);
grid_addBars(spec_rsq, 'h_scale', h.scale);
set(gcf, 'Visible','on');  
clear spec spec_rqs


% Optionally show maps of spectra, see bbci_bet_analyze_csp_sellap.m


% -- ERD/ERS --
name= sprintf('ERD-ERS for [%g %g] Hz', opt.band);
fig_set(figno_offset + 2, 'name',name, 'set', {'Visible','off'});
erd= proc_selectChannels(cnt_flt, requ_clab);
erd= proc_laplacian(erd);
erd= proc_envelope(erd, 'ms_msec', 200);
erd= cntToEpo(erd, mrk_all, opt.visu_ival);
erd= proc_baseline(erd, [], 'trialwise',0);
erd_rsq= proc_r_square_signed(proc_selectClasses(erd, classes));

h= grid_plot(erd, mnt, opt_grid);
grid_markIval(opt.ival);
grid_addBars(erd_rsq, 'h_scale', h.scale);
set(gcf, 'Visible','on');  
clear erd erd_rsq;


% Optionally show maps of ERD, see bbci_bet_analyze_csp_sellap.m



%% --- Feature extraction ---
%

BC_result.ival= opt.ival;
BC_result.band= opt.band;
BC_result.filt_b= filt_b;
BC_result.filt_a= filt_a;
BC_result.clab_csp= opt.clab_csp;
BC_result.sel_clab= sel_clab;

bbci.signal.clab= BC_result.clab;

epo= cntToEpo(cnt_flt, mrk2, BC_result.ival, 'clab',bbci.signal.clab);
clear cnt_flt
[fv, sel_w]= proc_selectChannels(epo, opt.clab_csp);

if isequal(opt.patterns,'auto'),
  [fv, csp_w, la, A]= proc_csp_auto(fv, 'patterns',opt.nPatterns);
else
  [fv, csp_w, la, A]= proc_csp3(fv, 'patterns',opt.nPatterns);
end
fig_set(figno_offset + 4, 'name', sprintf('CSP %s vs %s', classes{:}));
opt_scalp_csp= strukt('colormap', cmap_greenwhitelila(31));
if isequal(opt.patterns,'auto'),
  plotCSPanalysis(fv, mnt, csp_w, A, la, opt_scalp_csp, ...
                  'row_layout',1, 'title','');
else
  plotCSPanalysis(fv, mnt, csp_w, A, la, opt_scalp_csp, ...
                  'mark_patterns', opt.patterns);
end

fv_csp= proc_variance(fv);
fv_csp= proc_logarithm(fv_csp);

if ~isequal(opt.patterns,'auto'),
  fv_csp.x= fv_csp.x(opt.patterns,:);
  csp_w= csp_w(:,opt.patterns);
end
spat_w= [lap_w, sel_w*csp_w];
BC_result.csp_w= csp_w;
BC_result.spat_w= spat_w;
BC_result.feature= proc_flaten(proc_appendChannels(fv_lap, fv_csp));

bbci.signal.proc= {{@online_linearDerivation, BC_result.spat_w}, ...
                   {@online_filt, BC_result.filt_b, BC_result.filt_a}};

bbci.feature.ival= [-750 0];
bbci.feature.fcn= {@proc_variance, @proc_logarithm};

% The classifier is only trained on the 3 selected Laplacian channels.
% In order to be able switch to other channels during the feedback
% (bbci_adaptation_csp_plus_lap) features from all Laplacian channels are
% calculated. Here we insert components with 0 weight into the classifier to
% disregard the non-selected Laplacian channels.
idx_activelap= sort(chanind(fv_lap, sel_clab));
idx_csp= size(lap_w,2)+1:size(spat_w,2);
idx_active= [idx_activelap, idx_csp];
feat= BC_result.feature;
feat.x= BC_result.feature.x(idx_active,:);
bbci.classifier.C= trainClassifier(feat, opt.model);
w_tmp= bbci.classifier.C.w;
bbci.classifier.C.w= zeros(size(spat_w,2), 1);
bbci.classifier.C.w(idx_active)= w_tmp;
% Store some more information in the classifier, which is required to
% used 'bbci_adaptation_csp_plus_lap'.
bbci.classifier.model= opt.model;
bbci.classifier.fv_buffer= ...
    copy_struct(BC_result.feature, 'x','clab','y','className');
bbci.classifier.fv_buffer.idx_active= idx_active;
bbci.classifier.fv_buffer.idx_csp= idx_csp;
bbci.classifier.opt= copy_struct(opt, 'area', 'nlaps_per_area');

bbci.adaptation.fcn= @bbci_adaptation_csp_plus_lap;

bbci.quit_condition.marker= 255;


%% BB: this validation takes the selection of 3 laplacian filters
%% which have been selected from the complete data set!
opt_xv= strukt('sample_fcn',{'chronKfold',8}, ...
               'std_of_means',0, ...
               'verbosity',0, ...
               'progress_bar',0);
[loss,loss_std]= xvalidation(BC_result.feature, opt.model, opt_xv);
bbci_log_write(data, 'CSP global: %4.1f +/- %3.1f', 100*loss, 100*loss_std);
proc_logvar= 'fv= proc_variance(fv); fv= proc_logarithm(fv);';
proc= struct('memo',  {'spat_w'});
proc.sneakin= {'sel_clab',sel_clab, 'clab_csp',opt.clab_csp};
proc.apply= ['fv= proc_linearDerivation(fv, spat_w); ' ...
             proc_logvar];
if isequal(opt.patterns,'auto'),
  proc.train= ['[fv_lap,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
               '[fv,sel_w]= proc_selectChannels(fv, clab_csp); ' ...
               '[fv,csp_w]= proc_csp_auto(fv, ' int2str(opt.nPatterns) ');' ...
               'fv= proc_appendChannels(fv_lap, fv); ' ...
               'spat_w= [lap_w, sel_w*csp_w]; ' proc_logvar];
  [loss,loss_std]= xvalidation(epo, opt.model, opt_xv, 'proc',proc);
  bbci_log_write(data, 'CSP/LAP auto inside: %4.1f +/- %3.1f', ...
                 100*loss, 100*loss_std);
else
  proc.train= ['[fv_lap,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
               '[fv,sel_w]= proc_selectChannels(fv, clab_csp); ' ...
               '[fv,csp_w]= proc_csp3(fv, ' int2str(opt.nPatterns) '); ' ...
               'fv= proc_appendChannels(fv_lap, fv); ' ...
               'spat_w= [lap_w, sel_w*csp_w]; ' proc_logvar];
  [loss,loss_std] = xvalidation(epo, opt.model, opt_xv, 'proc',proc);
  bbci_log_write(data, 'CSP/LAP inside: %4.1f +/- %3.1f', ...
                 100*loss,100*loss_std);
  if ~isequal(opt.patterns, 1:2*opt.nPatterns),
    % The CSP filter that was calculated 'globally' on all samples is used
    % in the cross validation to select 'similar' filters
    proc.sneakin= cat(2, proc.sneakin, {'global_w',csp_w});
    proc.train= ['[fv_lap,lap_w]= proc_laplacian(fv, ''clab'',sel_clab); ' ...
                 '[fv,sel_w]= proc_selectChannels(fv, clab_csp); ' ...
                 '[fv,csp_w]= proc_csp3(fv, ''patterns'',global_w, ' ...
                     '''selectPolicy'',''matchfilters''); ' ...
                 'fv= proc_appendChannels(fv_lap, fv); ' ...
                 'spat_w= [lap_w, sel_w*csp_w]; ' proc_logvar];
    [loss,loss_std]= xvalidation(epo, opt.model, opt_xv, 'proc',proc);
    bbci_log_write(data, 'CSP/LAP selPat: %4.1f +/- %3.1f', ...
                   100*loss, 100*loss_std);
  end
end
mean_loss(ci)= loss;
std_loss(ci)= loss_std;

clear fv*

BC_result.figure_handles= figno_offset + [1 2 4];
data.result(ci)= BC_result;
cfy_fields= {'signal', 'feature', 'classifier', 'quit_condition'};
bbci_all(ci)= copy_struct(bbci, cfy_fields);

end  %% for ci  (class combinations)


%% --- Choose best binary combination of classes (if required) ---
%

nComb= size(class_combination,1);
if nComb > 1,
  data.all_results= data.result;
  [dmy, bi]= min(mean_loss + 0.1*std_loss);
  bbci= copy_fields(bbci, bbci_all(bi), cfy_fields);
  data.result= data.all_results(bi);
  data.result.class_selection_loss= [mean_loss; std_loss];
  bbci_log_write(data, sprintf('\nCombination <%s> vs <%s> chosen.\n', ...
                               data.result.classes{:}));
  % if there exist an artifact rejection figure include it in the list 
  if ismember(3, get(0, 'Children')),
    data.result.figure_handles= [data.result.figure_handles 3];
  end
    
  % minimize figures of not-chosen class combinations
  others= setdiff(1:nComb, bi);
  h_other_figs= cat(2, data.all_results(others).figure_handles);
  set(h_other_figs, 'Visible','off');
end

%% Store settings
data.figure_handles= data.result.figure_handles;
