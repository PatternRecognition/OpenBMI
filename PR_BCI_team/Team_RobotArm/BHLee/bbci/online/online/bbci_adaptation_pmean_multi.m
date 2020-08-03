function [cls,bbci]= bbci_adaptation_pmean_multi(cls, bbci, ts)
%[cls,bbci] = bbci_adaptation_pmean(cls, bbci, <varargin>)
%
% This function allows an unsupervised adaptation of the bias of LDA.
% It is called by adaptation.m, which is called by bbci_bet_apply.m
%
% Technique see Vidaurre et al 2008 (Graz paper).
%
% bbci.adaptation options specific for adaptation_pmean:
%   .verbose - 0: no output, 1: little output, 2: each bias change is
%      reported; default: 1
% bbci.adaptation should have the following fields for this to work:
%   .running - if this is 1, the adaptation process is going on.

persistent curr_mrk curr_feat window_counter running restart last_ts end_of_adaptation_marker
persistent ts_trialstart

if  ~isfield(bbci.adaptation, 'ix_adaptive_cls')
    warning('bbci.adaptation.ix_adaptive_cls is not specified, assuming all cl to be adaptive!' )
    bbci.adaptation.ix_adaptive_cls = 1
end
if  length(bbci.adaptation.ix_adaptive_cls) ~= 1
    warning('bbci.adaptation.ix_adaptive_cls has length unequal 1, first value is chosen!' )
    if  length(bbci.adaptation.ix_adaptive_cls) == 0
        bbci.adaptation.ix_adaptive_cls = 1
    else
        bbci.adaptation.ix_adaptive_cls =     bbci.adaptation.ix_adaptive_cls(ixAdapt)
    end
end

ixAdapt = bbci.adaptation.ix_adaptive_cls;

if ischar(ts) & strcmp(ts,'init')
    running = false;
    restart = true;
    last_ts= ts;
    return
end

if bbci.adaptation.running & (~running|isempty(running)) & restart,
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initial case
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [bbci.adaptation, isdefault]= ...
        set_defaults(bbci.adaptation, ...
        'UC', 0.03,...
        'delay', 250, ...
        'offset', 500,...
        'adaptation_ival', [], ...
        'mrk_start', [1 2 5],...
        'mrk_end', [11,12,21,22,23,24,25,51,52,53], ...
        'mrk_noupdate',{},...
        'load_tmp_classifier', 0, ...
        'verbose', 1);
    if isempty(bbci.adaptation.adaptation_ival),
        warning('use should use ''adaptation_ival''');
        bbci.adaptation.adaptation_ival= bbci.adaptation.offset + [bbci.adaptation.delay 0];
    else
        bbci.adaptation.delay= 0;
        bbci.adaptation.offset= 0;
    end
    if iscell(bbci.adaptation.mrk_start),
        warning('mrk_start should be a vector, not a cell array');
        bbci.adaptation.mrk_start= [bbci.adaptation.mrk_start{:}];
    end
    curr_mrk = [];
    curr_feat= zeros(size(cls(ixAdapt).C.w));
    tmp_classifier_loaded= 0;
    end_of_adaptation_marker= [];
    if bbci.adaptation.load_tmp_classifier,
        if exist([bbci.adaptation.tmpfile '.mat'], 'file'),
            load(bbci.adaptation.tmpfile, 'cls');
            tmpd= dir(bbci.adaptation.tmpfile);
            tmp_classifier_loaded= 1;
            if bbci.adaptation.verbose,
                fprintf('[adaptation_pmean:] classifier loaded from %s with date %s\n', ...
                    bbci.adaptation.tmpfile, tmpd.date);
            end
        else
            if bbci.adaptation.verbose,
                fprintf('[adaptation_pmean:] tmp classifier not found: %s\n', bbci.adaptation.tmpfile);
            end
        end
    end
    if ~tmp_classifier_loaded,
        if isfield(cls(ixAdapt), 'pmean'),
            if bbci.adaptation.verbose,
                fprintf('[adaptation_pmean:] Starting with already adapted classifier\n');
            end
        else
            cls(ixAdapt).pmean= mean(cls(ixAdapt).C.mean, 2);
            if bbci.adaptation.verbose,
                fprintf('[adaptation_pmean:] Starting with fresh classifier\n');
            end
        end
    end
    window_counter = 0;
    running = true;
    if bbci.adaptation.verbose,
        disp('[adaptation_pmean:] Adaptation started.');
        fprintf('temp classifier will be saved as %s\n', bbci.adaptation.tmpfile);
    end
    fprintf('\n[adaptation_pmean:] adaptation parameters:\n');
    bbci.adaptation
    fprintf('\n');
end

if ~bbci.adaptation.running & running,
    disp('Adaptation was stopped from the GUI')
    running = false;
    %if isfield(bbci,'update_port') & ~isempty(bbci.update_port)
    %  send_data_udp(bbci.gui_machine,bbci.update_port,...
    %                    double(['{''bbci.adaptation.running'',0}']));
    %else
    %  warning('No update port defined!');
    %end
end

if ~running,
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% later case: see if marker is in the queue.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(cls(ixAdapt),'offset')
    [toe,timeshift]= adminMarker('query', [last_ts-ts 0]);
    fprintf('bbci.adaptation.offset ignored!\n');  %% Just to check
else
    [toe,timeshift]= adminMarker('query', [last_ts-ts 0]);
end
last_ts= ts;

if isempty(curr_mrk),
    % not inside update interval.
    ind_startoftrial = intersect(bbci.adaptation.mrk_start, toe);
    ind_quit = intersect([bbci.adaptation.mrk_noupdate{:}], toe);
    if ~isempty(ind_startoftrial),
        if bbci.adaptation.verbose,
            fprintf('[adaptation_pmean:] Trigger received: %s\n', vec2str(toe));
        end
        % this starts a new trial.
        curr_mrk= 1;
        %    find_ind= find(ind_startoftrial(ixAdapt)==toe);
        %    store_ts= ts + timeshift(find_ind);
        toe= setdiff(toe, ind_startoftrial);
        ts_trialstart= ts;
    end
    if ~isempty(ind_quit)
        if bbci.adaptation.verbose,
            fprintf('[adaptation_pmean:] Adaptation stopped.\n');
        end
        running = false;
        restart = false;
        return
    end
    toe= [];
end

if isempty(curr_mrk)
    % not inside an update window. Just quit.
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if inside update window: average the feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ts-ts_trialstart >= bbci.adaptation.adaptation_ival{ixAdapt}(1) & ts-ts_trialstart <= bbci.adaptation.adaptation_ival{ixAdapt}(2),
    fn = cls(ixAdapt).fv;
    new_feat = getFeature('apply',fn,0);
    curr_feat= curr_feat + new_feat.x;
    window_counter = window_counter+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if at the end marker: put feature into fv.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isempty(toe)
    ind_endoftrial= intersect(bbci.adaptation.mrk_end, toe);
    if ~isempty(ind_endoftrial),
        end_of_adaptation_marker= ind_endoftrial;
    else
        % no marker recognized.
        if bbci.adaptation.verbose>1,
            fprintf('[adaptation_pmean:] Marker not used in adaptation: %s\n', ...
                vec2str(toe));
        end
    end
end

if ~isempty(end_of_adaptation_marker) & ...
        ts-ts_trialstart > bbci.adaptation.adaptation_ival{ixAdapt}(2),
    if bbci.adaptation.verbose,
        fprintf('[adaptation_pmean:] Endmarker: %d  (wc: %d)\n', ...
            end_of_adaptation_marker, window_counter);
    end
    if window_counter>0,
        % adapt bias by PMean method
        curr_feat= curr_feat/window_counter;
        cls(ixAdapt).pmean= (1-bbci.adaptation.UC)*cls(ixAdapt).pmean + ...
            bbci.adaptation.UC*curr_feat;
        cls(ixAdapt).C.b= -cls(ixAdapt).C.w' * cls(ixAdapt).pmean;
        writeClassifierLog('adapt', ts, cls(ixAdapt));
        save(bbci.adaptation.tmpfile, 'cls');
    end

    curr_mrk= [];
    curr_feat = zeros(size(cls(ixAdapt).C.w));
    window_counter= 0;
    end_of_adaptation_marker= [];
    if bbci.adaptation.verbose,
        fprintf('[adaptation_pmean:] new bias: %.3f\n', cls(ixAdapt).C.b);
    end
end