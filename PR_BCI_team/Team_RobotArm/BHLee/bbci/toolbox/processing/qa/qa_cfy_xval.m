

function [cfy H] = qa_cfy_xval(epo_orig,cfy_classes,varargin)
%
% USAGE:   
%   [cfy H] = qa_cfy_xval(epo_orig,cfy_classes,varargin)
%
% Perform a cross-validation 
%
% IN:       epo_orig            -   Epoched data structure   
%           cfy_classes         -   Cell array of strings defining which trials
%                                   should be used for classification.
%                                   Strings can be:
%                                   'TN': true negatives
%                                   'FN': false negatives
%                                   'TP': true  positives
%                                   'FP': false positives
%                                   'all' is equal to {'TN','FN','TP','FP'}
%           varargin            -   optional arguments (see code for details)
%
% OUT:      'cfy'-struct:
%                   .auc                 -   area-under-curve values for each stimulus condition 
%                   .loss_xv             -   xval-loss for each stimulus condition 
%                   .xval_out            -   xval-classfier output for each stimulus condition     
%                   .labels              -   corresponding class labels for each stimulus condition     
%                   .ivals               -   classificaion intervals used for each stimulus condition    
%                   .hit_idx_cfy         -   indices of hits (TP) of the classification 
%                                            w.r.t. the input structure 'epo_orig'
%                   .miss_idx_cfy        -   indices of misses (FN) of the classification 
%                                            w.r.t. the input structure 'epo_orig'
%                   .signif              -   significance test output (corresponding to opt.signif_levels) 
%                                            of the classifier discrimination for each stimulus levels 
%
%           H: figure handle (if opt.visualize=1)
%
%
%  Simon Scholler, 2011
%


warning off

% check if function is called through matgrid
if nargin==1 && iscell(epo_orig)
   varargin = epo_orig{4:end};
   cfy_classes = epo_orig{2};
   epo_orig = epo_orig{1};
end


opt= propertylist2struct(varargin{:});
opt = set_defaults(opt, 'nCSPfilts', 4);
opt = set_defaults(opt, ...
                 'min_num_trials',30, ...            % minimum number of trials per class
                 'equal_classsize',0, ...
                 'classification', 'ERP', ...        % temporal: 'ERP' or spectral: 'ERD'
                 'signif_levels', [0.01 0.05], ...   % significance levels on classification (cf. val_ranksumtest.m)
                 'visualize', 0, ...                 % plot figures?                 
                 'nConds', length(epo_orig.className), ...                 
                 ... %%% ERP OPTS %%%
                 'cfy_ivals', 100:100:700, ...
                 'xval_params_erp', {'RLDAshrink', 'xTrials', [1 10], 'verbosity',0, 'save_proc_params',{'cfy_ivals'}}, ... 
                 ...%'xval_params_erp', {'RLDAshrink', 'sample_fcn', 'leaveOneOut', 'verbosity',0}, ... 
                 'find_cfy_ival', [0 epo_orig.t(end)], ...  % only used when cfy_ivals is a number (input to select_time_intervals())
                 'sign', 0, ... % only used when cfy_ivals is a number (input to select_time_intervals())
                 'discr_fcn', 'sgn_r2', ...                 
                 ... %%% ERD OPTS %%%
                 'xval_params_erd', {'RLDAshrink', 'xTrials', [1 10], 'verbosity',0, 'save_proc_params','csp_w'}, ...                 
                 'proc_train', ['[fv,csp_w]= proc_csp_auto(fv, ' int2str(opt.nCSPfilts) '); ' ...
                                'fv= proc_variance(fv); ' ...
                                'fv= proc_logarithm(fv);'], ...
                 'proc_apply', ['fv= proc_linearDerivation(fv, csp_w); ' ...
                                'fv= proc_variance(fv); ' ...
                                'fv= proc_logarithm(fv);']);

                            
% Init return variables
cfy.auc= NaN(1,opt.nConds);
cfy.xval_out=  cell(1,opt.nConds);
cfy.labels = cell(1,opt.nConds);
cfy.ivals =  cell(1,opt.nConds);
cfy.loss_xv= NaN(1,opt.nConds);
cfy.loss_std= NaN(1,opt.nConds);
cfy.signif =  cell(1,opt.nConds);
cfy.miss_idx_cfy = [];
cfy.hit_idx_cfy = [];

if opt.visualize
    H = figure;
else
    H = [];
end

if isequal(cfy_classes,'all')
   cfy_classes = {'TN','FP','FN','TP'};
end

%% Select classification subset (e.g. TP vs TN, or (FN,TP) vs TN)
[epo orig_idx cfy_classes] = qa_get(epo_orig,cfy_classes);

%% Determine the comparison class (commonly, this is the baseline class)
epo1 = qa_get(epo_orig,cfy_classes{1});
epo2 = qa_get(epo_orig,cfy_classes{2});
if length(epo1.className)==1
    cclass = epo1.className{:};  
    cc = cfy_classes{1}; 
    vc = cfy_classes{2};
elseif length(epo2.className)==1
    cclass = epo2.className{:};
    cc = cfy_classes{2}; 
    vc = cfy_classes{1};
else
    error('Both classes in ''cfy_classes'' are present in more than one stimulus class');
end


%% Determine classes that have a sufficient number of trials. These will
%% then be classified pairwise against the comparison class
N_cw = sum(epo.y,2);
valid_classes = find(arrayfun(@(x) opt.min_num_trials <= x, N_cw));
% remove the comparison class
valid_classes(strmatch(cclass,epo.className)) = [];
n_vc = length(valid_classes);

% Get subject code
sbj= epo.title(1:find(epo.title=='_',1)-1);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% LQ-LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%

for c = valid_classes'    
        
    N = sum(epo.y(c,:));    
    stimlev= epo.className{c};
    c_orig = strmatch(stimlev,epo_orig.className); % class index in epo_orig
    epo_cfy= proc_selectClasses(epo, {cclass, stimlev});

    %% Reduce xval-dataset if wanted
    if opt.equal_classsize
        % careful: idx_miss and idx_hq is subsampled to only contain x trials of misses and of HQ
        % with x=opt.min_num_trials
        idx_c1= find(epo_cfy.y(1,:));  % class 1
        idx_c2= find(epo_cfy.y(2,:));  % class 2
        idx_c1_sub= idx_c1(randperm(length(idx_c1)));
        idx_c1_sub= idx_c1_sub(1:opt.min_num_trials);
        idx_c2_sub= idx_c2(randperm(length(idx_c2)));
        idx_c2_sub= idx_c2_sub(1:opt.min_num_trials);
        idx_cfy = union(idx_c1_sub, idx_c2_sub);
        epo_cfy = proc_selectEpochs(epo, idx_cfy);  
    end                        
    
    
    %% Compute cfy-features
    switch opt.classification
        
        case 'ERD'
            fv = epo_cfy;
            fv.proc= struct('memo', 'csp_w');
            fv.proc.train = opt.proc_train;
            fv.proc.apply = opt.proc_apply;
            
            %% XVALIDATION
            [cfy.loss_xv(c_orig), cfy.loss_std(c_orig), out_test, memo] = xvalidation(fv, opt.xval_params_erd{:});
            
            cfy.csp_filter{c_orig} = memo.csp_w;
            
        case 'ERP'
            
            switch opt.discr_fcn
                case 'sgn_r2'
                    epo_r= proc_r_square_signed(epo_cfy);
                    xval_discr_fcn = 'proc_r_square_signed(fv)';
                case 'r2'
                    epo_r= proc_r_square(epo_cfy);
                    xval_discr_fcn = 'proc_r_square(fv)';
                case 'roc'
                    epo_r= proc_rocAreaValues(epo_cfy);
                    xval_discr_fcn = 'proc_rocAreaValues(fv)';
            end
            
            if opt.visualize
                subplot(2,n_vc,c)
                fraction = round(100*N/sum(epo_orig.y(c,:)));
                suptitle([epo_cfy.className{2} '(' vc ')_{' int2str(fraction) '%} vs. HQ']);
            end
            
            if ~isscalar(opt.cfy_ivals)
                ival_cfy = opt.cfy_ivals;
                if opt.visualize
                    imagesc(epo_r.t, 1, epo_r.x'); colorbar
                    colormap(cmap_posneg(51))
                    set(gca, 'CLim',[-1 1]*max(abs(epo_r.x(:))));
                end                
                fv= proc_jumpingMeans(epo_cfy, ival_cfy);                
            else
                fv = epo_cfy;
                fv.proc= struct('memo', 'cfy_ivals');
                fv.proc.train = ['cfy_ivals = select_time_intervals(' xval_discr_fcn ',''nIvals'',' int2str(opt.cfy_ivals) ',''sort'', 1,' ...
                                '''ival_pick_peak'', [' int2str(opt.find_cfy_ival) '],''sign'',' int2str(opt.sign) ', ''visualize'', 0);' ...
                                 'fv = proc_jumpingMeans(fv, cfy_ivals);'];
                fv.proc.apply = 'fv = proc_jumpingMeans(fv, cfy_ivals);';  
            end
            %% XVALIDATION
            try
                [cfy.loss_xv(c_orig), cfy.loss_std(c_orig), out_test, memo] = xvalidation(fv, opt.xval_params_erp{:});
            catch
                fv.proc.train = ['cfy_ivals = select_time_intervals(' xval_discr_fcn ',''nIvals'',' int2str(opt.cfy_ivals) ',''sort'', 1,' ...
                                '''ival_pick_peak'', [' int2str(opt.find_cfy_ival) '],''sign'', 0, ''visualize'', 0);' ...
                                 'fv = proc_jumpingMeans(fv, cfy_ivals);'];
                [cfy.loss_xv(c_orig), cfy.loss_std(c_orig), out_test, memo] = xvalidation(fv, opt.xval_params_erp{:});
            end
            cfy.cfy_ivals{c_orig} = memo.cfy_ivals;           
    end        
   
    
    %% Determine which instances have been classified as hits and misses
    % Stimulus class hit-miss indices
    cidx = find(epo.y(c,:));
    sl_idx = find(fv.y(2,:));
    midx = find(out_test(sl_idx)<0);
    hidx = find(out_test(sl_idx)>0);    
    cfy.miss_idx_cfy = [cfy.miss_idx_cfy orig_idx(cidx(midx))]; % map back to indices of the input epo struct
    cfy.hit_idx_cfy = [cfy.hit_idx_cfy orig_idx(cidx(hidx))];
    % Baseline class hit-miss indices
    cidx = find(epo.y(1,:));
    bl_idx = find(fv.y(1,:));
    midx = find(out_test(bl_idx)>0);
    hidx = find(out_test(bl_idx)<0);    
    cfy.bl_miss_idx_cfy{c_orig} = orig_idx(cidx(midx)); % map back to indices of the input epo struct
    cfy.bl_hit_idx_cfy{c_orig} = orig_idx(cidx(hidx));
    
    %% Plot ROC curve
    if opt.visualize
        subplot(2,n_vc,n_vc+c)        
        [roc, roc_auc]= val_rocCurve(fv.y, out_test, 'plot', 1);
        title(sprintf('%s, %s', sbj, stimlev), 'FontSize', 12, 'FontWeight', 'b');
        text(0.5,0.2, sprintf('AUC= %.4f %s', roc_auc, sigstr),'FontSize', 16);
    else
        [roc, roc_auc]= val_rocCurve(fv.y, out_test, 'plot', 0);
    end
        
    %% save the computed data    
    cfy.auc(c_orig)= roc_auc;
    cfy.xval_out{c_orig}= out_test;
    cfy.labels{c_orig}= fv.y;
    cfy.ivals{c_orig}= opt.cfy_ivals;
    cfy.confusionMatrix = val_confusionMatrix(fv.y,out_test);
    % test for significance
    cfy.signif{c} = val_ranksumtest(fv.y, out_test, opt.signif_levels);

    clear('epo_cfy', 'epo_r', 'idx*', 'fv', 'out_test', 'roc_auc', 'roc', ...
        'ival_cfy');
end; % c

if opt.visualize
    colorbars('equalize');
end

