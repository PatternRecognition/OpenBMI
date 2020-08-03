function [bbci, data]= bbci_calibrate_ERP_Speller(bbci, data)
%BBCI_CALIBRATE_ERP_SPELLER - Calibrate online system for ERP Spellers
%
%This function is called by bbci_calibrate 
%(if BBCI.calibate.fcn is set to @bbci_calibrate_ERP_Speller).
%Via BBCI.calibrate.settings, the details can be specified, se below.
%
%Synopsis:
% [BBCI, DATA]= bbci_calibrate_ERP_Speller(BBCI, DATA)
% 
%Arguments:
%  BBCI -  the field 'calibrate' holds parameters specific to ERP Spellers.
%  DATA -  holds the calibration data
%  
%Output:
%  BBCI - Updated BBCI structure in which all necessary fields for
%     online operation are set, see bbci_apply_structures.
%  DATA - As input but added some information of the analysis that might
%     be reused in a second run
%
%BBCI.calibrate.settings may include the following parameters:
%  ...to.be.filled.in...
%  'cue_markers': Markers that trigger the classification of an ERP,
%          default: [11:16,21:26,31:36,41:46]

% 11-2011 Benjamin Blankertz


opt= bbci.calibrate.settings;
opt= set_defaults(opt, ...
                  'disp_ival', [-200 800], ...
                  'ref_ival', [-200 0], ...
                  'cfy_clab', {'not','E*','Fp*','AF*','A*'}, ...
                  'cfy_ival', 'auto', ...
                  'cfy_ival_pick_peak', [100 700], ...
                  'control_per_stimulus', 0, ...
                  'model', 'RLDAshrink', ...
                  'create_figs', 1);

default_grd= sprintf(['scale,FC3,FC1,FCz,FC2,FC4,legend\n' ...
                     'C5,C3,C1,Cz,C2,C4,C6\n' ...
                     'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
                     'P7,P5,P3,Pz,P4,P6,P8\n' ...
                     'PO7,PO3,O1,Oz,O2,PO4,PO8']);

default_crit= strukt('maxmin', 100, ...
                     'clab', {'EOG*','F9,10','Fp*'}, ...
                     'ival', [100 800]);
default_crit.ival(1)= max(opt.disp_ival(1), default_crit.ival(1));
default_crit.ival(2)= min(opt.disp_ival(2), default_crit.ival(2));

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'clab', '*', ...
                 'cfy_maxival', [], ...
                 'reject_artifacts', 1, ...
                 'reject_artifacts_opts', {}, ...
                 'reject_channels', 1, ...
                 'reject_eyemovements', 0, ...
                 'reject_eyemovements_crit', default_crit, ...
                 'grd', default_grd, ...
                 'clab_erp', {'CPz','PO7'}, ...
                 'clab_rsq', {'CPz','PO7'}, ...
                 'target_dist', 0, ...
                 'nSequences', 5, ...
                 'nClasses', 6, ...
                 'cue_markers', [11:16,21:26,31:36,41:46], ...
                 'mrk2feedback_fcn', []);
% add possibly missing fields to opt.reject_eyemovements_crit
opt.reject_eyemovements_crit= ...
  set_defaults(opt.reject_eyemovements_crit, default_crit);

[opt, isdefault]= ...
  opt_overrideIfDefault(opt, isdefault, ...
                        'clab_rsq', opt.clab_erp);

clear fv*
opt_scalp_erp= defopt_scalp_erp2('colorOrder', [0.9 0 0.9; 0.4 0.4 0.4]);
opt_scalp_r= defopt_scalp_r2;

mnt= mnt_setGrid(data.mnt, opt.grd);

if ~data.isnew && isfield(data, 'result'),
  previous= data.result;
else
  previous= struct;
end

if sum(data.mrk.y(2,:))>3000 % if requires (more that 3000 non-target epochs) reduce the number of datapoints to prevent memory problems 
my_frac = 3000 / sum(data.mrk.y(2,:)) ;
ixNT = find(data.mrk.y(2,:));
ix_selectNT = rand(1,length(ixNT)) < my_frac;
ixNT = ixNT(find(ix_selectNT));
ixT = find(data.mrk.y(1,:));
ixMerge = sort([ixNT ixT])
data.mrk = mrk_selectEvents(data.mrk, ixMerge);
warning(['TOO MUCH DATA!!! taking only ' num2str(round(my_frac*100)) '% of the nonTargets (randomly chosen)!'])
end

if isfield(bbci.calibrate.settings, 'band')
    if ~isempty(bbci.calibrate.settings.band)
%         [filt_b,filt_a]= butter(5, bbci.calibrate.settings.band/data.cnt.fs*2);
        [filt_b,filt_a]= cheby2(5, 20, bbci.calibrate.settings.band/data.cnt.fs*2);
        data.cnt = proc_filt(data.cnt, filt_b, filt_a);
%         data.cnt = proc_filtfilt(data.cnt, filt_b, filt_a);
    end
end


BC_result= [];
BC_result.mrk= data.mrk;

BC_result.clab= data.cnt.clab(chanind(data.cnt, opt.clab));
BC_result.cfy_clab= data.cnt.clab(chanind(data.cnt, opt.cfy_clab));
% The following figures are always generated
if opt.create_figs,
    data.figure_handles= [1 2 4 6];
end

if opt.target_dist,
  [BC_result.mrk, BC_result.selected_trials]= ...
      mrk_selectTargetDist(BC_result.mrk, opt.target_dist);
else
  BC_result.selected_trials= NaN;
end


%% --- Artifact rejection (trials and/or channels) based on variance criterion
%
flds= {'reject_artifacts', 'reject_channels', ...
       'reject_artifacts_opts', 'target_dist', 'clab'};
if data.isnew || ~isfield(data, 'previous_settings') || ...
      ~fieldsareequal(opt, data.previous_settings, flds),
  BC_result.rejected_trials= NaN;
  BC_result.rejected_clab= NaN;
  if opt.reject_artifacts | opt.reject_channels,
    if opt.create_figs, 
        fig_set(5, 'name','Artifact rejection');         
        data.figure_handles(end+1)= 5;
    end
    [mk_clean , rClab, rTrials]= ...
        reject_varEventsAndChannels(data.cnt, BC_result.mrk, ...
                                    opt.disp_ival, ...
                                    'clab',BC_result.clab, ...
                                    'do_multipass', 1, ...
                                    'verbose', 1, ...
                                    'visualize', opt.create_figs, ...
                                    opt.reject_artifacts_opts{:});
    if opt.reject_artifacts,
      bbci_log_write(data, 'Rejected: %d trial(s).', length(rTrials));
      BC_result.mrk= mrk_chooseEvents(BC_result.mrk, 'not',rTrials);
      BC_result.rejected_trials= rTrials;
    end
    if opt.reject_channels,
      bbci_log_write(data, 'Rejected channels: <%s>', vec2str(rClab));
      BC_result.rejected_clab= rClab;
    end
  else
    % Avoid confusion with old figure from previous run
    close_if_exists(5);
  end
  if iscell(BC_result.rejected_clab),   %% that means rejected_clab is not NaN
    cidx= find(ismember(BC_result.clab, BC_result.rejected_clab));
    BC_result.clab(cidx)= [];
    cidx= find(ismember(BC_result.cfy_clab, BC_result.rejected_clab));
    BC_result.cfy_clab(cidx)= [];
  end
else
  result_flds= {'rejected_trials', 'rejected_clab', 'clab', 'cfy_clab'};
  BC_result= copy_fields(BC_result, previous, result_flds);
end


%% --- Segmentation and baselining ---
%
epo= cntToEpo(data.cnt, BC_result.mrk, opt.disp_ival, ...
              'clab', BC_result.clab);

%% Rejection of eyemovements based on max-min criterium 
%
if opt.reject_eyemovements && opt.reject_eyemovements_crit.maxmin>0,
  [epo, iArte]= ...
      proc_rejectArtifactsMaxMin(epo, ...
                                 opt.reject_eyemovements_crit.maxmin, ...
                                 'clab', opt.reject_eyemovements_crit.clab, ...
                                 'ival', opt.reject_eyemovements_crit.ival, ...
                                 'verbose',1);
  BC_result.mrk= mrk_chooseEvents(BC_result.mrk, 'not',iArte);
  BC_result.eyemovement_trials= iArte;
else
  BC_result.eyemovement_trials= NaN;
end

epo= proc_baseline(epo, opt.ref_ival, 'channelwise', 1);


%% --- Plot r^2 matrix and select intervals if requested ---
%
%epo_r= proc_r_square_signed(epo);
epo_r= proc_channelwise(epo, 'r_square_signed');
epo_r= rmfields(epo_r, {'V','p'});
epo_r.className= {'sgn r^2 ( T , NT )'};  %% just make it shorter
if opt.create_figs, fig_set(6, 'name','r^2 Matrix'); end
if isempty(opt.cfy_ival) || isequal(opt.cfy_ival, 'auto'),
  [BC_result.cfy_ival, nfo]= ...
      select_time_intervals(epo_r, 'visualize', opt.create_figs, 'visu_scalps', opt.create_figs, ...
                            'sort', 1, 'intersample_timing', 1, ...
                            'clab_pick_peak',opt.cfy_clab, ...
                            'ival_pick_peak', opt.cfy_ival_pick_peak, ...
                            'ival_max', opt.cfy_maxival);
  BC_result.cfy_ival= visutil_correctIvalsForDisplay(BC_result.cfy_ival, 'fs',epo.fs);
  bbci_log_write(data.log.fid, 'Selected time intervals:');
  bbci_log_write(data.log.fid, '  [%g %g] ms', BC_result.cfy_ival');
else
  BC_result.cfy_ival= opt.cfy_ival;
  BC_result.cfy_ival= visutil_correctIvalsForDisplay(BC_result.cfy_ival, 'fs',epo.fs);
  if opt.create_figs, visualize_score_matrix(epo_r, BC_result.cfy_ival); end
end


%% --- Visualize ERPs ---
%
if opt.create_figs,
    fig_set(1, 'name','ERP - grid plot', 'set',{'Visible','off'});
    H= grid_plot(epo, mnt, defopt_erps, 'colorOrder',opt_scalp_erp.colorOrder);
    %grid_markIval(BC_result.cfy_ival);
    if isfield(H, 'scale'),
      grid_addBars(epo_r, 'h_scale',H.scale);
    else
      grid_addBars(epo_r);
    end
    set(gcf,  'Visible','on');

    fig_set(2, 'name','ERP - scalp maps');
    H= scalpEvolutionPlusChannel(epo, mnt, opt.clab_erp, BC_result.cfy_ival, ...
                                 opt_scalp_erp);
    grid_addBars(epo_r);

    if isempty(opt.clab_rsq) | isequal(opt.clab_rsq,'auto'),
      opt.clab_rsq= unique_unsort({nfo.peak_clab}, 4);
    end
    fig_set(4, 'name','ERP - r^2 scalp maps');
    scalpEvolutionPlusChannel(epo_r, mnt, opt.clab_rsq, BC_result.cfy_ival, ...
                                opt_scalp_r);
    clear epo*
end


%% --- Feature extraction ---
%
BC_result.ref_ival= opt.ref_ival;
BC_result.nSequences= opt.nSequences; %% Future extension: select by simulation

bbci.signal.clab= BC_result.cfy_clab;

if isfield(bbci.calibrate.settings, 'band') %set filter if required!
    if ~isempty(bbci.calibrate.settings.band)
        online_sampl_rate = data.cnt.fs; %extract sampling rate from bbci-strukt 
        [filt_b,filt_a]= cheby2(5, 20, bbci.calibrate.settings.band/online_sampl_rate*2);
        bbci.signal.proc= {{@online_filt, filt_b, filt_a}};
    end
end


if isempty(opt.ref_ival)% no baselining
        bbci.feature.proc= {{@proc_jumpingMeans, BC_result.cfy_ival}}; 
        min_t = min(BC_result.cfy_ival(:));
else
    bbci.feature.proc= {{@proc_baseline, BC_result.ref_ival, 'beginning_exact'}, ...
                    {@proc_jumpingMeans, BC_result.cfy_ival}};
    min_t =  min(BC_result.ref_ival(1), min(BC_result.cfy_ival(:)));
end
bbci.feature.ival= [min_t BC_result.cfy_ival(end)];

if opt.control_per_stimulus,
  bbci.control.fcn= @bbci_control_ERP_Speller_binary;
else
  bbci.control.fcn= @bbci_control_ERP_Speller;
end
if isempty(opt.mrk2feedback_fcn),
  mrk2feedback_fcn= @(x)x;
else
  mrk2feedback_fcn= opt.mrk2feedback_fcn;
end
bbci.control.param= {struct('nClasses',opt.nClasses, ...
                            'nSequences',BC_result.nSequences, ...
                            'mrk2feedback_fcn', mrk2feedback_fcn)};
bbci.control.condition.marker= opt.cue_markers;
if ~isfield(bbci, 'quit_condition') || ~isfield(bbci.quit_condition, 'marker'),
    bbci.quit_condition.marker = 255;
end

fv= cntToEpo(data.cnt, BC_result.mrk, bbci.feature.ival, ...
             'clab',bbci.signal.clab);
fv= bbci_calibrate_evalFeature(fv, bbci.feature);

bbci.classifier.C= trainClassifier(fv, opt.model);

opt_xv= strukt('xTrials', [1 10], ...
               'loss','rocArea', ...
               'verbosity', 0);

[loss,dum,outTe]= xvalidation(fv, opt.model, opt_xv);
me= val_confusionMatrix(fv, outTe, 'mode','normalized');
bbci_log_write(data.log.fid, ['ROC-loss: %.1f%%  |  ' ...
                    'Correct Hits: %.1f%%,  Correct Miss: %.1f%%'], ...
               100*loss, 100*me(1,1), 100*me(2,2));

if isfield(bbci.calibrate, 'early_stopping_fnc'),
    data.cls_out = outTe;
    bbci = bbci.calibrate.early_stopping_fnc(bbci, data);
end
           
data.feature= fv;
data.result= BC_result;
data.result.rocloss = 100*loss;
data.result.me = me;