function [ loss, loss_std, me, info ] = stdERPclassification( epo, varargin )
%STDERPCLASSIFICATION method that makes an classification (target vs
%non-targets) for a given epo
%
% [ loss, loss_std, me, info ] = stdERPclassification( epo, varargin )
%
% INPUT
% epo
% opt.
%     ivals          default: []
%     ival_coverage
%                   default: strukt('sampling_deltas', [20, 60], 'sampling_ivals', [80,350; 360, 800])
%     heuristicsConstraints
%                   default:[]
%     model         default: 'FDshrink'
%     loss_fcn      default classwiseNormalized
%     n_folds       default: 5, ...
%     n_shuffles    default: 5, ...
%     verbose       default: 1 ...
%
% SEEALSO stdERPanalysis stdERPplots
%
% Johannes Hoehne 10.2011 j.hoehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'ivals',  [], ... %if not specified, ivals are within as specified in ival_coverage (DEFAULT) or the constraint is used with parameters specifies in opt.heuristicsConstraints
    'ival_coverage', strukt('sampling_deltas', [20, 60], 'sampling_ivals', [80,350; 360, 800]), ... %
    'heuristicsConstraints', [], ... % { {-1, [100 300], {'I#','O#','PO7,8','P9,10', 'F5-6'}, [100 300]}, ... {1, [250 600], {'P3-4','CP3-4','C3-4'}, [250 600]}       }, ...
    'r2_smoothing_width', 50, ... %only needed if heuristic constrains were set
    'model', 'FDshrink', ...
    'loss_fcn', 'classwiseNormalized', ...
    'n_folds', 5, ...
    'n_shuffles', 5, ...
    'verbose', 1, ...
    'opt_xv',[] ...
    );

defopt_xv= strukt('xTrials', [opt.n_shuffles opt.n_folds], 'loss', opt.loss_fcn, 'out_timing', 0, 'progress_bar', 0, ...
    'proc', strukt('train', '[fv,paraMean]=proc_subtractMean(fv); [fv,paraNorm]=proc_normalize(fv);', ...
    'apply', '[fv]=proc_subtractMean(fv, paraMean); [fv]=proc_normalize(fv, paraNorm);', ...
    'memo', {'paraMean', 'paraNorm'} ));

opt.opt_xv = set_defaults(opt.opt_xv, defopt_xv);

if isempty(opt.ivals)
    if isstruct(opt.ival_coverage)
        disp('taking dense intervals! might need a lot of memory!')
        epo_sampling_ivals = opt.ival_coverage.sampling_ivals;
        epo_sampling_deltas = opt.ival_coverage.sampling_deltas;
        opt.ivals = [];
        for k=1:size(epo_sampling_ivals,1)
            t_lims = epo_sampling_ivals(k,:);
            dt = epo_sampling_deltas(k);
            sampling_points = t_lims(1):dt:t_lims(2);
            tmp_ivals = [sampling_points' - dt/2, sampling_points' + dt/2];
            opt.ivals = [opt.ivals; tmp_ivals];
        end
    elseif iscell(opt.heuristicsConstraints)
        epo_r = proc_rocAreaValues(epo);
        epo_r = proc_movingAverage(epo_r, opt.r2_smoothing_width, 'method', 'centered'); %smoothing to prevent artifactual local minima
        [opt.ivals, nfo]= ...
            select_time_intervals(epo_r, 'visualize', 0, 'visu_scalps', 0, ...
            'clab',{'not','E*'}, ...
            'constraint', opt.heuristicsConstraints, 'nIvals', length(opt.heuristicsConstraints));
        warning('Intervals are selected with heuristic!')
    else
        error('ival, ival_coverage or heuristicsConstraints has to be specified')
    end
    
end

features= proc_jumpingMeans(epo, opt.ivals);
features = proc_flaten(features);
%     [features, opt.meanOpt] = proc_subtractMean();
%     [features, opt.normOpt] = proc_normalize(features);


if nargout > 3 %  finally , a trained model shall be delivered
    if (isfield(opt.model, 'param'))
        % select_model notwendig, da noch freie Hyperparameter zu bestimmen sind
        [opt.model, loss, loss_std, P, outTe, ms_memo] = select_model(features, opt.model, opt.opt_xv);
        disp('model selection finished.');
    else
        % xvalidation reicht aus
        [loss,loss_std,outTe, xval_memo] = xvalidation(features, opt.model.classy, defopt_xv);
    end
    info = {};
    info.C= trainClassifier(features,opt.model);
    info.selfApplied_out = applyClassifier(features, opt.model, info.C);
    info.features = features;
    info.opt = opt;
    info.xval_out = outTe;
    info.ivals = opt.ivals;
    if exist('xval_memo')
        info.xval_memo = xval_memo;
    end
    if exist('ms_memo')
        info.ms_memo=ms_memo;
    end
else
    % loss and maybe loss_std are required, but no trained model
    if (isfield(opt.model, 'param'))
        % select_model notwendig, da noch freie Hyperparameter zu bestimmen sind
        [opt.model, loss, loss_std, P, outTe, ms_memo] = select_model(features, opt.model, opt.opt_xv);
        disp('model selection finished.');
    else
        % xvalidation reicht aus
        [loss,loss_std,outTe, xval_memo] = xvalidation(features, opt.model.classy, defopt_xv);
    end
end

me= val_confusionMatrix(features, outTe, 'mode','normalized');
if opt.verbose
    
    remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
    disp(remainmessage);
    sprintf('intervals taken:\n')
    disp(opt.ivals)
end


end

