
function [Cw cfy_ivals] = qa_cfy_getLDAfilters(epo,nFilters,varargin)
%
% USAGE:   [Cw cfy_ivals] = qa_cfy_getLDAfilters(epo,nFilters,varargin)
%
% Perform a cross-validation 
%
% IN:       epo                 -   epo-struct containing exactly 2 classes.    
%           nFilters            -   Number of LDA filters that should be computed. 
%                                   Corresponds to the number of intervals
%                                   that are selected by select_time_intervals.
%           varargin            -   optional arguments (see code for details) 
%
% OUT:      Cw                  -   Matrix of size (nChannels x nFilters),
%                                   i.e. each column is one LDA filter
%           cfy_ivals           -   the time intervals on which the LDAs is
%                                   computed: cfy_ivals(n,:) corresponds to 
%                                   LDA filter Cw(:,n)
%
% Simon Scholler, 2010/2011
%

opt= propertylist2struct(varargin{:});
opt = set_defaults(opt, ...    
                 'cfy_classes', {'TN','TP'}, ...
                 'select_ival_opts', {'nIvals', nFilters, 'find_cfy_ival', [50 epo.t(end)],'sort',1, 'visualize',0}, ...
                 'discr_fcn', 'sgn_r2');  % discrimination function
                 

             
%% Select type of classification (e.g. TP vs TN)
if ~isempty(opt.cfy_classes)
    epo = qa_get(epo,opt.cfy_classes);
end
if length(epo.className)~=2
    error('Input epo-struct must have exactly 2 classes')
end

%% Compute cfy-features
switch opt.discr_fcn
    case 'sgn_r2'
        epo_r= proc_r_square_signed(epo);
    case 'r2'
        epo_r= proc_r_square(epo);
    case 'roc'
        epo_r= proc_rocAreaValues(epo);
end

if ~isfield(opt.select_ival_opts, 'nIvals')
   opt.select_ival_opts = propertylist2struct(opt.select_ival_opts{:});
   opt.select_ival_opts.nIvals = nFilters; 
else
    if opt.select_iva_opts.nIvals~=nFilters
        error('Input arguments don''t match.')
    end
end

% select discriminative time interval
cfy_ivals = select_time_intervals(epo_r, opt.select_ival_opts);

if size(cfy_ivals,1) < nFilters
    warning(['Not enough intervals found by select_time_interval-heuristic.' ... 
            'Now using equally sized and spaced intervals. Press return to continue'])
    pause
    borders = round(linspace(opt.find_cfy_ival(1), opt.find_cfy_ival(2), nFilters+1))';
    cfy_ivals = [borders(1:end-1) borders(2:end)];
end

Cw = [];
for f = 1:nFilters
    C = trainClassifier(proc_jumpingMeans(epo, cfy_ivals(f,:)),'RLDAshrink');
    Cw = [Cw C.w];
end


