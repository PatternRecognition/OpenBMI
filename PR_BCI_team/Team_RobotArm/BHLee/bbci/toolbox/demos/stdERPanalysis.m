function [ epo, epo_r, ivals, cnt, misc ] = stdERPanalysis( file, stimDef, varargin )
%% this methods loads data of a typical ERP experiment and performs a
%% standard analysis with the following steps:
% 
% SYNOPSIS:
% [ epo, epo_r, ivals, cnt, misc ] = stdERPanalysis( file, stimDef, varargin );
%
% EXAMPLE:
%fname = 'VPhbb_11_10_13/Audit_stim_screening_cond_4*';
% use all ''default'' settings
% [epo epo_r] = stdERPanalysis(fname, {[11:19] [1:9]; 'Target' ,'Non-Target'});
%
% use P300design
% [epo, epo_r] = stdERPanalysis(fname, {[11:19] [1:9]; 'Target' ,'Non-Target'}, 'P300design', [9,14]);
%
% define filters yourself
% [epo epo_r] = stdERPanalysis(fname, {[11:19] [1:9]; 'Target' 'Non-Target'}, 'plotting', 1, 'hp_filt', [.4, .2, 3, 30], 'ref_ival', [-150 0], 'artifactRejection_var', 0, 'lp_filt', [17 25 3 50]);
% 
% INPUT
% file   filename of raw-EEG files
% stimDef definition of classes
% varargin ... see below
% 
%     (1) load raw data and 
%        apply filters (opt.lp_filt & opt.hp_filt as boundaries) 
%     (2) select channels (opt.clab)
%     (3) rereference (opt.commonReference)
%     (4) build mrk structure
%     (5) artifact rejection based on variance criterion (opt.artifactRejection_var)
%         --> channels might be disregarded in this step
%     (6) segmentation into epo (opt.disp_ival)
%     (7) artifact rejection based on max-min criterion (opt.crit_minmax & opt.crit_ival)
%     (8) compute discriminability - ROC or r-squared (opt.useROC)
%     (9) plot the ERPs (opt.plotting)
%     (10) look for discriminative ivals
%
% SEEALSO stdERPplots stdERPclassification
% 
% Johannes Hoehne 10.2011 j.hoehne@tu-berlin.de


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'fs', 100, ...
    'plotting', 0, ...
    'plotArtifacts', 0, ...
    'useROC', 0, ...
    'commonReference', [], ...
    'disp_ival', [-150 800], ...
    'ref_ival', [-150 0], ... %if [] --> no referencing 
    'clab', {'*'}, ...  %reduce channels to
    'crit_ival', [100 550], ...
    'lp_filt', [40 49 3 50], ...
    'hp_filt', [], ... % [.4, .1, 3, 30]
    'notch_filt', 0, ... %flag to apply a 50Hz notch filter
    'heuristicsConstraints',  { {-1, [100 250], {'*'}, [100 250]}, ...
    {1, [250 400], {'FC3-4','P3-4','CP3-4','C3-4'}, [250 400]}, {1, [350 650], {'P3-4','CP3-4','C3-4'}, [350 650]}        }, ... %only used if ivals in argout
    'artifactRejection_var', 1, ...
    'reduce2scalpChannels', 0, ...
    'func_mrkodef', 'mrkodef_general_oddball', ...
    'crit_minmax', 80, ...
    'crit_minmax_chan', 'EOG*', ...
    'EOG_chanDef', {}, ... %might be used to define EOG, example: {'EOGv','EOGvu','Fp2'; 'EOGh','F9','F10'}
    'StimulusDelay', 0, ... % Delay in ms for each stimulus. Positive delay: marker is earlier than actual stimulus! Useful if there is a known latency for stimulation (such as pygame-latency for auditory stimuli ~50ms)
    'P300design', []); % (optional) for P300_infos in epo number of Classes, (default []), either vector [nClasses, nRepetitions] or strukt: strukt('nClasses', 6, 'nRepetitions', 15);
     


iArte = []; rClab = []; rTrials = [];

opt_mrk= strukt('respDef',{},  'stimDef', stimDef);

%% (1) load raw data do filtering 
hdr= eegfile_readBVheader(file);
if ~isempty(opt.lp_filt)   
    Wps= [opt.lp_filt(1) opt.lp_filt(2)]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2), opt.lp_filt(3), opt.lp_filt(4));
    [filt.b, filt.a]= cheby2(n, 50, Ws);
else
    filt = [];
end

[cnt, mrk_orig]= eegfile_readBV(file, 'fs',opt.fs, 'filt',filt);

% highpass the cnt
if ~isempty(opt.hp_filt)
            Wps = [opt.hp_filt(1) opt.hp_filt(2)]/cnt.fs*2;
            [n, Wn] = buttord(Wps(1), Wps(2), opt.hp_filt(3), opt.hp_filt(4));
            [filt.b, filt.a]= butter(n, Wn, 'high');
            cnt = proc_filtfilt(cnt, filt.b, filt.a);
end

if opt.notch_filt
    cnt = proc_filtnotch(cnt);
end

%% (2) select channels
cnt = proc_selectChannels(cnt, opt.clab);

%% (3) rereferencing if required
if ~isempty(opt.commonReference)
    cnt = proc_commonAverageReference(cnt, opt.commonReference, '*');
end

if ~isempty(opt.EOG_chanDef)
cnt = proc_bipolarEOGcustom(cnt, opt.EOG_chanDef);
end

%% (4) build mrk structure
mrk= feval(opt.func_mrkodef, mrk_orig, opt_mrk);
mrk_beforeArtifactRejection = mrk;
if  ~isempty(opt.P300design) 
    if  isnumeric(opt.P300design) && length(opt.P300design) ==2
        mrk = mrk_addInfo_P300design(mrk, opt.P300design(1), opt.P300design(2));
    elseif isstruct(opt.P300design)
        mrk = mrk_addInfo_P300design(mrk, opt.P300design.nClasses, opt.P300design.nRepetitions, opt.P300design);
    end
end
if opt.StimulusDelay ~= 0
    mrk = mrkutil_shiftEvents(mrk, opt.StimulusDelay);
end

    

%% (5) artifact rejection based on variance criterion
if opt.artifactRejection_var
    chan_before = cnt.clab;
    if opt.plotArtifacts
        fig_set(1);
    end    
    [mrk_tmp, rClab, rTrials]= reject_varEventsAndChannels(cnt, mrk, opt.disp_ival, ...
        'visualize', opt.plotArtifacts);
    while ~isempty(rClab)
        %When there are channels to remove, then we look for artifactual trials
        %once more!
        cnt = proc_selectChannels(cnt, chanind(cnt.clab, 'not', rClab));
        [mrk_tmp, rClab, rTrials]= reject_varEventsAndChannels(cnt, mrk, opt.disp_ival, ...
            'visualize', opt.plotArtifacts);
    end
    mrk = mrk_tmp;
    
    if length(chan_before)-length(cnt.clab) > 0
        fprintf('%d channels (%s)and %d trials removed due to variance criterion \n', ...
            length(chan_before)-length(cnt.clab), cell2mat(setdiff(chan_before, cnt.clab)), length(rTrials));
    else
        fprintf('no channels and %d trials removed due to variance criterion \n', ...
            length(rTrials));
    end
end

if opt.reduce2scalpChannels
    cnt = proc_selectChannels(cnt, intersect(cnt.clab, scalpChannels));
end

%% (6) segmentation into epo
epo= cntToEpo(cnt, mrk, opt.disp_ival);
if  ~isempty(opt.ref_ival)
    epo= proc_baseline(epo, opt.ref_ival);
end
%epo= proc_detrend(epo);

%% (7) artifact rejection based on max-min criterion
if opt.crit_minmax>0,
    crit= struct('maxmin', opt.crit_minmax);
    epo_crit= proc_selectIval(epo, opt.crit_ival);
    iArte= find_artifacts(epo_crit, opt.crit_minmax_chan, crit);
    fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
        length(iArte), crit.maxmin);
    clear epo_crit
    epo= proc_selectEpochs(epo, 'not',iArte);
end





%% (8) compute discriminability 
if nargout > 1 && size(epo.y,1) > 1
    if opt.useROC
        epo_r= proc_rocAreaValues(epo);
        epo_r.className= {'AUC ( t , nt )'};  %% just make it shorter
    else
        epo_r= proc_r_square_signed(epo);
        epo_r.className= {'sgn r^2 ( t , nt )'};  %% just make it shorter
    end


%% (9) plot the ERPs
    if opt.plotting
      stdERPplots(epo, epo_r)
    end

%% (10) look for discriminative ivals
    if nargout > 2
        epo_r_smooth = proc_movingAverage(epo_r, 50, 'method', 'centered'); %smoothing to prevent artifactual local minima
        [opt.ivals, nfo]= ...
            select_time_intervals(epo_r_smooth, 'visualize', 0, 'visu_scalps', 0, ...
            'clab',{'not','E*', 'Ref'}, ...
            'constraint', opt.heuristicsConstraints, 'nIvals', length(opt.heuristicsConstraints));
        ivals= visutil_correctIvalsForDisplay(opt.ivals, 'fs',epo.fs);
    end
end

if nargout > 3
    misc = strukt('mrk', mrk, 'mrk_orig', mrk_orig, 'mrk_beforeArtifactRejection', mrk_beforeArtifactRejection, 'maxmin_rejTrials', iArte, 'var_rejTrials', rTrials, 'var_rejChans', rClab, 'hdr', hdr)
end
end

