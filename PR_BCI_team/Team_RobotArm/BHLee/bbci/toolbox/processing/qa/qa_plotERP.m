

function [H H_discr erp_hits erp_misses] = qa_plotERP(epo,varargin)
%
% USAGE:  [H H_discr erp_hits erp_misses] = qa_plotERP(epo,ival,varargin)
%
% Plot ERP scalpplots and channel separately for each stimulus condition, 
% and separately according to field epo.detected
%
% IN:       epo                 -   Epoched data strcture
%           varargin            -   optional arguments (see code for details)
%
% OUT:      H                   -   figure handle for ERP figure
%           H_discr             -   figure handle for figure which depicts
%                                   the discrimination of the stimulus
%                                   conditions with the baseline condition
%           erp_hits            -   ERP data structure for trials with
%                                   epo.detected==1
%           erp_misses          -   ERP data structure for trials with
%                                   epo.detected==0     
%
% Simon Scholler, 2011
%


opt= propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                 'min_epochs', 10, ... % minimal number of epochs per average, if #epochs<opt.minepochs, the average will not be shown
                 'visuchan', 'CPz', ... 
                 'ival', epo.t([1 end]), ...
                 'ival_scalps', epo.t(1):200:epo.t(end), ...
                 'opt_scalp', my_opt_scalp, ...
                 'discr_fcn', 'sgn_r2', ...  % discrimination function
                 'visualize', 1, ...         % plot figures?
                 'fig2subplot', 0, ...     % merge single figures into one
                 'titles', {'Hits','Misses'}, ...
                 'label_nTrials', 1, ...   % show #Trials per class in label. if 0, classwise percentages will be shown                 
                 'compare_to_BL', 0);      % compare classes against baseline? 

             
set(0,'DefaultAxesFontSize',14)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Default Options
mnt = getElectrodePositions(epo.clab);
figsize = [0.5 0 0.5 1];

R = qa_getDetectionData(epo);
if opt.label_nTrials
    R_hits = R(:,2).*R(:,3);
    R_misses = R(:,3) - R(:,2).*R(:,3);
else
    R_hits = round(R(:,2)).*100;
    R_misses = 1 - R_hits;
end
% Get subject code
sbj= epo.title(1:find(epo.title=='_',1)-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%  ERP-ANALYSIS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

epo_hits = proc_selectEpochs(epo, 'not', find(epo.detected==0),'removevoidclasses',0);
epo_misses = proc_selectEpochs(epo, 'not', find(epo.detected==1),'removevoidclasses',0);
for c = 1:length(epo.className)
    % rename classes
    epo_hits.className{c} = [epo_hits.className{c} '-h' num2str(R_hits(c),'%.0f')];
    epo_misses.className{c} = [epo_misses.className{c} '-m' num2str(R_misses(c),'%.0f')];    
    % remove class in the plots if class has less than opt.min_epochs trials 
    cidx_hits = find(epo_hits.y(c,:)); 
    cidx_misses = find(epo_misses.y(c,:)); 
    if length(cidx_hits)<opt.min_epochs        
        epo_hits.x(:,:,cidx_hits) = NaN;
    end
    if length(cidx_misses)<opt.min_epochs        
        epo_misses.x(:,:,cidx_misses) = NaN;
    end
end


%%% ERP_STIM %%%
erp_hits = proc_average(epo_hits);
erp_misses = proc_average(epo_misses);
colorOrder = get_colorOpt(erp_hits);
%sl_max = proc_selectClasses(erp_hits,erp_hits.className{end});
sl_li = proc_jumpingMeans(erp_hits,opt.ival_scalps);
li = max(abs(sl_li.x(:)));
if isnan(li)
    sl_li = proc_jumpingMeans(erp_misses,opt.ival_scalps);
    li = max(sl_li.x(:));
end
cLim = [-li li];
hasdata = @(epo) ~all(isnan(epo.x(:)));

if opt.visualize
    H_hits = figure('Units','normalized','Position',figsize); axis off
    if hasdata(erp_hits)
        scalpEvolutionPlusChannel(erp_hits, mnt, opt.visuchan, opt.ival_scalps, ...
            opt.opt_scalp, 'colorOrder', colorOrder,'cLim', cLim);
        suptitle(['\bf ' opt.titles{1}])
    end
    H_misses = figure('Units','normalized','Position',figsize); axis off
    if hasdata(erp_misses)
        scalpEvolutionPlusChannel(erp_misses, mnt, opt.visuchan, opt.ival_scalps, ...
            opt.opt_scalp, 'colorOrder', colorOrder, 'cLim', cLim);
        suptitle(['\bf ' opt.titles{2}])
    end
    
    if opt.fig2subplot
        position = [0 0 0.5 1; 0.5 0 0.5 1];
        H = fig2subplot([H_hits H_misses], 'positions', position, 'deleteFigs', 1);
    else
        H = {H_hits H_misses};
    end
    
    
    if opt.compare_to_BL
        %% r2_hitVsMiss:
        % if exist('r2_hm','var')
        %     clear r2_hm
        % end
        % for c = 1:length(epo_hits.className);
        %    hits = proc_selectClasses(epo_hits, epo_hits.className{c});
        %    misses = proc_selectClasses(epo_misses, epo_misses.className{c});
        %    if ~isempty(hits.x) && ~isempty(misses.x)
        %        r2 = proc_r_square_signed(proc_appendEpochs(hits,misses));
        %        r2.className{1} = r2.className{1}(10:end-2);
        %        idx = findstr(r2.className{1},' , ');
        %        r2.className{1} = r2.className{1}([1:idx-1 idx+1 idx+3:end]);
        %        if ~exist('r2_hm','var')
        %            r2_hm = r2;
        %        else
        %            r2_hm = proc_appendEpochs(r2_hm,r2);
        %        end
        %    else
        %        if isempty(hits.x)
        %           hits.className = {[epo_hits.className{c} ' , ' epo_misses.className{c}]};
        %           hits.y = 1;
        %           hits.x = NaN(size(hits.x,1), size(hits.x,2), 1);
        %           if ~exist('r2_hm','var')
        %               r2_hm = hits;
        %           else
        %               r2_hm = proc_appendEpochs(r2_hm,hits);
        %           end
        %        elseif isempty(misses.x)
        %           misses.className = {[epo_hits.className{c} ' , ' epo_misses.className{c}]};
        %           misses.y = 1;
        %           misses.x = NaN(size(misses.x,1), size(misses.x,2), 1);
        %           if ~exist('r2_hm','var')
        %               r2_hm = misses;
        %           else
        %               r2_hm = proc_appendEpochs(r2_hm,misses);
        %           end
        %        end
        %    end
        % end
        
        %% r2_hitVsBL:
        nClasses = length(epo_hits.className);
        r2_hhq = proc_selectClasses_keepVoids(epo_hits,2:nClasses);
        r2_hhq = proc_appendEpochs(r2_hhq, proc_selectClasses(epo,1));  % add BL trials
        voids = find(sum(r2_hhq.y,2)==0);   % void classes
        voids_classnames = r2_hhq.className(voids);
        if strcmpi(opt.discr_fcn,'roc')
            r2_hhq = proc_rocAreaValues(r2_hhq,'multiclass_policy','all-against-last', 'ignoreNaN',1);
        else
            r2_hhq = proc_r_square_signed(r2_hhq,'multiclass_policy','all-against-last', 'tolerate_nans',1);
            flatclass = strmatch(['sgn r^2( ' epo_hits.className{end} ' , flat )'],r2_hhq.className);
            if ~isempty(flatclass)    % correct for strange behaviour of proc_r_square_signed
                r2_hhq = proc_selectClasses(r2_hhq,'not',r2_hhq.className{flatclass});
            end
        end
        
        for f = 1:length(r2_hhq.className)
            if strcmpi(opt.discr_fcn,'roc')
                r2.className{f} = r2_hhq.className{f}(11:end-3);
            else
                r2_hhq.className{f} = r2_hhq.className{f}(10:end-2);
            end
            idx = findstr(r2_hhq.className{f},' , ');
            r2_hhq.className{f} = r2_hhq.className{f}([1:idx-1 idx+1 idx+3:end]);
        end
        
%         if ~strcmpi(opt.discr_fcn,'roc')
%             % add void classes (were removed by proc_r_square_signed)
%             for v = length(voids):-1:1
%                 r2_hhq = proc_appendVoidClass(r2_hhq,voids_classnames{v});
%                 r2_hhq.x = r2_hhq.x(:,:,[length(r2_hhq.className) 1:length(r2_hhq.className)-1]);
%                 r2_hhq.className = r2_hhq.className([length(r2_hhq.className) 1:length(r2_hhq.className)-1]);
%             end
%         end
        
        r2_hhq = proc_appendVoidClass(r2_hhq,epo.className{1});
        r2_hhq.x = r2_hhq.x(:,:,[nClasses 1:nClasses-1]);
        r2_hhq.className = r2_hhq.className([nClasses 1:nClasses-1]);
        
        
        %% r2_missesVsBL:
        nClasses = length(epo_misses.className);
        r2_mhq = proc_selectClasses_keepVoids(epo_misses,2:nClasses);
        r2_mhq = proc_appendEpochs(r2_mhq, proc_selectClasses(epo,1));  % add BL trials
        voids = find(sum(r2_mhq.y,2)==0);   % void classes
        voids_classnames = r2_mhq.className(voids);
        if strcmpi(opt.discr_fcn,'roc')
            r2_mhq = proc_rocAreaValues(r2_mhq,'multiclass_policy','all-against-last', 'ignoreNaN',1);
        else
            r2_mhq = proc_r_square_signed(r2_mhq,'multiclass_policy','all-against-last', 'tolerate_nans',1);
            flatclass = strmatch(['sgn r^2( ' epo_hits.className{end} ' , flat )'],r2_mhq.className);
            if ~isempty(flatclass)    % correct for strange behaviour of proc_r_square_signed
                r2_mhq = proc_selectClasses(r2_mhq,'not',r2_mhq.className{flatclass});
            end
        end
        
        for f = 1:length(r2_mhq.className)
            if strcmpi(opt.discr_fcn,'roc')
                r2.className{f} = r2_mhq.className{f}(11:end-3);
            else
                r2_mhq.className{f} = r2_mhq.className{f}(10:end-2);
            end
            idx = findstr(r2_mhq.className{f},' , ');
            r2_mhq.className{f} = r2_mhq.className{f}([1:idx-1 idx+1 idx+3:end]);
        end
        
        % if ~strcmpi(opt.discr_fcn,'roc')
        %     % add void classes (were removed by proc_r_square_signed)
        %     for v = 1:length(voids)
        %         r2_mhq = proc_appendVoidClass(r2_mhq,voids_classnames{v});
        %
        %     end
        % end
        
        r2_mhq = proc_appendVoidClass(r2_mhq,epo.className{1});
        r2_mhq.x = r2_mhq.x(:,:,[nClasses 1:nClasses-1]);
        r2_mhq.className = r2_mhq.className([nClasses 1:nClasses-1]);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Print r2 plots %%%
        
        
        H_r2_hhq = figure('Units','normalized','Position',figsize); axis off       
        if hasdata(r2_hhq)
            colorOrder = get_colorOpt(r2_hhq);
            scalpEvolutionPlusChannel(r2_hhq, mnt, opt.visuchan, opt.ival_scalps, ...
                opt.opt_scalp, 'colorOrder', colorOrder);
            suptitle(['\bf ' opt.titles{1} ' vs. ' epo.className{1}])
        end
        
        H_r2_mhq = figure('Units','normalized','Position',figsize); axis off     
        if hasdata(r2_mhq)
            colorOrder = get_colorOpt(r2_mhq);
            scalpEvolutionPlusChannel(r2_mhq, mnt, opt.visuchan, opt.ival_scalps, ...
                opt.opt_scalp, 'colorOrder', colorOrder);
            suptitle(['\bf ' opt.titles{2} ' vs. ' epo.className{1}])
        end
        
        if opt.fig2subplot
            position = [0 0 0.5 1; 0.5 0 0.5 1];
            H_discr = fig2subplot([H_r2_hhq H_r2_mhq], 'positions', position, 'deleteFigs', 1);
        else
            H_discr = {H_r2_hhq H_r2_mhq};
        end
    end
end


%% Prepare hit and miss strutures for grand average
if nargout==0
    clear H H_discr erp_hits erp_misses   
elseif nargout>2
    for n = 1:length(erp_hits)
        erp_hits.x(find(isnan(erp_hits.x(:)))) = 0;
        erp_misses.x(find(isnan(erp_misses.x(:)))) = 0;
    end
end

set(0,'DefaultAxesFontSize',10)

