function [classy, E, S, P, out_test,ms_memo] = select_model(dat, model, varargin)
%[classy, loss, loss_std, params] = select_model(fv, model, <opt>)
%
% ToDo: clarify role of ms_memo and ms_memo.mesha as a return parameter with Janne. 
% (Michael, 2012_06_27)
%
%select model parameter(s) to minimize some loss function
%
% IN  fv - feature vectors, struct with
%               .x (data, nd array where last dim is nSamples) and
%               .y (labels [nClasses x nSamples], 1 means membership)
%               ('substructure' of data structure of epoched data)
%     model - two possibilities:
%             1. no struct (trivial case): a classifier with free parameters,
%                no selection, just call 'xvalidation'.
%             2. struct with the following entries:
%      .classy   - name of a classifier (string), or a cell array
%                  {classy_fcn, param1, ...}. Free parameters, which
%                  should be selected by 'select_model' are set to
%                  '*lin' or '*log', or are controlled by the field
%                  model.param (see there)
%      .xTrials  - [#shuffles, #folds]
%                  if a third value #samples is specified, for each trial
%                  a subset containing #samples is drawn from fv for each
%                  shuffle of the x-validation, default [3 10]
%                      you can use -1 as 3 component, e.g.:
%                      [3 10 -1] is translated to
%                      [3 10 round(9/10*"# of trials")]
%      .msDepth  - depth of recursion, default 1
%      .param    - (1) vector of values to be tested for the free parameter
%                   in model.classy (indicated by '*lin' or '*log', if
%                   necessary '*lin' are added to model.classy at the end).
%                   in recursion (.msDepth >1), select_model
%                   zooms in, using linear ('*lin') or logarithmic
%                   scaling ('*log').
%                  (2) struct array, one struct for each free parameter,
%                   with the following fields:
%                   .index - the place where the params must be set
%                            in the call of the train of the classifier
%                   .value - vector of values to be tested for the
%                            respective free parameter,
%                            default 0.05:0.1:0.95 for lin or -4:4 for log.
%                   .scale - 'lin' or 'log', default 'lin'
%                   .range - the range in which parameters are tested
%                            when doing recursion (.msDepth>1),
%                            default [0 1]  for lin, [-10 10] for log
%      .std_factor  - the goal function which select_model seeks to minimize
%                     is loss + loss_std * opt.std_factor,
%                     default 0 minimizes the loss regardless of its std
%      .std_consider- when opt.msDepth>1, model_selection further investigates
%                     the vicinity of all parameters that have a loss <=
%                       loss_min + loss_min_std * opt.std_consider,
%                     default 0 investigates only the vicinity of the best
%      opt:
%      .verbosity   - level of verbosity, default 1
%      .fixed_sets  - use a fixed selection of training/test sets throughout
%                     model selection, default 1
%      .divTr       - cell array of fixed train sets
%      .divTe       - cell array of fixed test sets
%      .estimate_ge - a function that estimates the generalization error
%                     of a classifier on a given data set (fv), format
%                     fcn(fv, classy, <opt>) like xvalidation (default)
%      furthermore the following fields of opt override, if present,
%      the respective fields of model:
%      'msDepth', 'xTrials', 'std_factor', 'range'
%      finally opt is passed to the opt.estimate_ge function.
%
% OUT classy   - cell array (or a string) of the classifier with the
%                best fixed params (maybe one is choose)
%     loss     - minimum loss (mean loss at selected parameter)
%     loss_std - standard deviation of the loss for the select parameter
%     params   - the best params
%     out_test - meant to be used when xvalidation is used to estimate
%                the generalisation error:
%                continuous classifier output for each x-val trial /
%                sample, a matrix of size [nClasses nSamples nTrials].
%                For classifiers with reject option: see below.

% For the 2nd argument 'model' for can also use a cell array of models.
% In that case a model selection is done for each model defined in the
% cell array, and finally the best of the selected models if chosen.
% Note: first model in the cell array must NOT be specified by a
% mere string. To avoid that you can put the string into a cell, e.g.
% instead of {'LDA', model_SVM} use {{'LDA'}, model_SVM}.
%
% SEE xvalidation, sampleDivisons
%
% Guido Dornhege (Guido.Dornhege@first.fhg.de)
% last update 07.01.04
% for questions or comments please contact me.

%                   params: structure (intern use for recursion).
%                           If this field exist, model.param will
%                           be ignored.
%                       index: a array with the order of indices
%                              where the params must be
%                       scale: cell array with lin and log (default lin)
%                       minim: array with minimal values (default 0
%                              if lin, -10 if log)
%                       maxim: array with maximal values (default 1
%                              if lin, 10 if log)
%                       param: cell array, in each cell is one
%                              parameter combination
%                       value: cell array with all possible values
%                              for each parameter


if ~isstruct(dat)
    error('the 1st argument must be a struct');
end
if ~isfield(dat,'x')
    error('the first argument must contain the data under the field x');
end
if ~isfield(dat,'y')
    error('the first argument must contain the labels under the field y');
end
if ~exist('model','var')
    error('no model as 2nd argument is specified');
end

ms_memo = []; % default output argument e.g. for the case of recursion where ms_memo is not used currently

%% select from a cell array of models
if iscell(model) & ~ischar(model{1}),
    min_ml= inf;
    for cc= 1:length(model),
        [classif, loss, loss_std, param, out_test]= ...
            select_model(dat, model{cc}, varargin{:});
        mloss= loss(1) + opt.std_factor*loss_std(1);
        if mloss<min_ml,
            min_ml= mloss;
            classy= classif;
            E= loss;
            S= loss_std;
            P= param;
        end
    end
    return;
end

default_scale= 'lin';

if length(varargin)==1 & isstruct(varargin{1}) & isfield(varargin{1},'depth'),
    opt= varargin{1};
else
    %% we are not in the recursion: do the setup
    if length(varargin)==1 & isreal(varargin{1}),
        opt.xTrials= varargin{1};
    else
        opt= propertylist2struct(varargin{:});
    end
    
    opt= set_defaults(opt, ...
        'verbosity', 1, ...
        'dsply_precision', 3, ...
        'fixed_sets', 1, ...
        'estimate_ge', 'xvalidation', ...
        'linvalue', 0.05:0.1:0.95, ...
        'linrange', [0 1], ...
        'logvalue', -4:4, ...
        'logrange', [-10 10], ...
        'er', [], ...
        'variance', [], ...
        'out_prefix', '', ...
        'dsply_plusminus', char(177));
    %                   'dsply_plusminus', '+/-');
    
    override_defaults= struct('msDepth', 1, ...
        'xTrials', [3 10], ...
        'sample_fcn', 'divisions', ...
        'std_factor', 0, ...
        'std_consider', 0);
    
    override_fields= fieldnames(override_defaults);
    for ii= 1:length(override_fields),
        fld= override_fields{ii};
        if ~isfield(opt, fld) | isempty(getfield(opt, fld)),
            if isstruct(model) & isfield(model,fld) & ~isempty(getfield(model,fld)),
                value= getfield(model, fld);
            else
                value= getfield(override_defaults, fld);
            end
            opt= setfield(opt, fld, value);
        end
    end
    
    if isequal(opt.sample_fcn, 'divisions'),
        opt.sample_fcn= {opt.sample_fcn, opt.xTrials};
    end
    if isfield(opt, 'xTrials'),
        opt= rmfield(opt, 'xTrials');
    end
    
    % model no struct
    if ~isstruct(model)
        classy= model;
        if nargout>1,
            [E, S, out_test]= feval(opt.estimate_ge, dat, model, opt);
            P= [];
        end
        return;
    end
    
end

% Prepare the indices for shuffles and folds
if (~isfield(opt,'divTr') | isempty(opt.divTr)) & opt.fixed_sets,
    [func, sample_params]= getFuncParam(opt.sample_fcn);
    sample_fcn= ['sample_' func];

    opt= set_defaults(opt, 'check_bidx_labels',1); % added by Michael 2012_06_27, Janne please re-check!
    [repIdx, eqcl]= xval_choose_repIdx(dat,opt.check_bidx_labels);

    [opt.divTr, opt.divTe]= feval(sample_fcn, dat.y(:,repIdx), sample_params{:});
    for ii= 1:length(opt.divTr), % shuffles
        for jj= 1:length(opt.divTr{ii}), % folds
            opt.divTr{ii}{jj}= eqcl(opt.divTr{ii}{jj});
            opt.divTe{ii}{jj}= eqcl(opt.divTe{ii}{jj});
        end
    end
end

params = [];
fmt= ['%.' int2str(opt.dsply_precision) 'f'];


% model.classy no cell-array
if ~iscell(model.classy), model.classy= {model.classy}; end

% if no model.params exist , a lot is to do
if ~isfield(model,'params') | isempty(model.params),
    
    % param no struct
    if ~isfield(model,'param'),
        error('you must specifiy param as field in the second argument');
    end
    if ~isstruct(model.param),
        V= model.param;
        model.param =[];
        model.param.index = getModelParameterIndex(model.classy);
        model.param.value = V;
        ip = length(model.param);
        for i= 1:ip,
            if model.param(i).index>length(model.classy) | ...
                    isempty(model.classy{model.param(i).index}),
                model.classy{model.param.index}= ['*' default_scale];
            end
            switch(model.classy{model.param(i).index})
                case '*log'
                    model.param(i).scale = 'log';
                case '*lin'
                    model.param(i).scale = 'lin';
                otherwise
                    error(['the format for model.param.scale is not known.' ...
                        'Only log and lin possible']);
            end
        end
    end
    
    % other default for param
    for ip = 1:length(model.param)
        if ~isfield(model.param(ip),'index') | isempty(model.param(ip).index)
            error('no index in model.param is specified');
        end
        if ~isfield(model.param(ip),'scale') | isempty(model.param(ip).scale)
            % The previous version 1.13 had a bug here, it always used
            % default_scale if scale was not provided in the param structure.
            % This had nice effects: User provides '*log' in the argument list,
            % but does not set the 'scale' field, hence select_model was using
            % linear scaling!
            % New code: set scale based on what the user provided in the
            % argument list
            if model.param(ip).index<=length(model.classy),
                switch(model.classy{model.param(ip).index})
                    case '*log'
                        model.param(ip).scale = 'log';
                    case '*lin'
                        model.param(ip).scale = 'lin';
                    otherwise
                        error('Unknown format for model.param.scale, only log and lin allowed');
                end
            else
                model.param(ip).scale = default_scale;
            end
        end
        % Just in case we need to extend the search range: sizes of the
        % intervals at the left and right end of the values to guess suitable
        % extensions
        values = sort(model.param(ip).value);
        if length(model.param(ip).value)>1,
            interv1 = values(2)-values(1);
            interv2 = values(end)-values(end-1);
        else
            interv1 = 0;
            interv2 = 0;
        end
        useDefaultRange = 0;
        switch(model.param(ip).scale)
            case 'lin'
                if ~isfield(model.param(ip), 'value') | ...
                        isempty(model.param(ip).value),
                    model.param(ip).value= opt.linvalue;
                end
                if ~isfield(model.param(ip),'range') | ...
                        isempty(model.param(ip).range)
                    model.param(ip).range= opt.linrange;
                    useDefaultRange = 1;
                end
            case 'log'
                if ~isfield(model.param(ip), 'value') | ...
                        isempty(model.param(ip).value),
                    model.param(ip).value= opt.logvalue;
                end
                if ~isfield(model.param(ip),'range') | ...
                        isempty(model.param(ip).range)
                    model.param(ip).range= opt.logrange;
                    useDefaultRange = 1;
                end
            otherwise
                error ('unknown scale');
        end
        % Extend the range if the given values already extend the given (or default)
        % range, otherwise depth search might run into trouble. Do this
        % extension silently if the user did not provide a range, otherwise complain
        if min(values) < model.param(ip).range(1),
            if ~useDefaultRange,
                warning('Range of values exceeds the provided search range. Now extending search range.');
            end
            model.param(ip).range(1) = min(values)-2*interv1;
        end
        if max(values) > model.param(ip).range(2),
            if ~useDefaultRange,
                warning('Range of values exceeds the provided search range. Now extending search range.');
            end
            model.param(ip).range(2) = max(values)+2*interv2;
        end
    end
    
    %translate to the right format (params)    
    for ip= 1:length(model.param),
        model.params.index(ip) = model.param(ip).index;
        model.params.scale{ip} = model.param(ip).scale;
        model.params.minim(ip) = model.param(ip).range(1);
        model.params.maxim(ip) = model.param(ip).range(2);
        model.params.value{ip} = sort(model.param(ip).value);
    end
    mesha = meshall(model.param(:).value);
    nVV = prod(size(mesha{1}));
    for i= 1:nVV,
        for j= 1:length(model.params.index),
            model.params.param{i}(j) = mesha{j}(i);
        end
    end
elseif ~isfield(opt,'depth') | isempty(opt.depth)
    % default for model.params
    n = length(model.params.index);
    for i= 1:n
        if ~isfield(model.params,'scale') | isempty(model.params.scale{i})
            model.params.scale{i} = scale;
        end
        if ~isfield(model.params,'minim') | isempty(model.params.minim(i))
            if strcmp(model.params.scale{i},'log')
                model.params.minim(i) = opt.logrange(1);
            else
                model.params.maxim(i) = opt.linrange(1);
            end
        end
        if ~isfield(model.params,'maxim') | isempty(model.params.maxim(i))
            if strcmp(model.params.scale{i},'log')
                model.params.minim(i) = opt.logrange(2);
            else
                model.params.maxim(i) = opt.linrange(2);
            end
        end
    end
    if ~isfield(model.params,'param') | isempty(model.params.param),
        error('model.params.param must be specified');
    end
    if ~isfield(model.params,'value') | isempty(model.params.value),
        for i= 1:n,
            model.params.value{i}(1) = model.params.param{1}(i);
            for ip= 2:length(model.params.param),
                c= find(model.params.param{ip}(i)<=model.params.value{i}(:));
                if isempty(c)
                    d= 1;
                else
                    d= max(c);
                end
                if ~(model.params.param{ip}(i) == model.params.value{i}(d)),
                    for l= length(model.params.value{i}):-1:1,
                        model.params.value{i}(l+1) = model.params.value{i}(l);
                    end
                    model.params.value{i}(d) = model.params.param{ip}(i);
                end
            end
        end
    end
end

% if opt.fixed_sets divTr, divTe must set in dat
if opt.fixed_sets  & (~isfield(opt,'divTr') | isempty(opt.divTr)),
    dat.divTr = opt.divTr;
    dat.divTe = opt.divTe;
end

% for depthsearch opt.depth has to be set
if opt.msDepth>1 & (~isfield(opt,'depth') | isempty(opt.depth))
    opt.depth = 1;
end

% a lot of changes in model are needed, therefore we make a copy
modell = model;

% important variables
nCombos= length(model.params.param);
nParam = length(model.params.index);

% some nice output
if opt.verbosity,
    if ~isfield(opt, 'depth'),
        if ~isempty(opt.out_prefix),
            fprintf('%s: ', opt.out_prefix);
        end
        fprintf('model selection ');
        %    if isfield(dat, 'title'),
        %      fprintf('for %s ', dat.title);
        %    end
        fprintf('using %s\n', toString(model.classy));
    end
    if isfield(opt,'depth')
        fprintf('Depthsearch %d\n', opt.depth);
    end
    if ~isfield(opt,'depth') | opt.depth==1,
        fprintf('Format parameter: ');
        expStr = '';
        if strcmp(model.params.scale{1},'log'), expStr = '10^'; end
        fprintf('%d: %sP -> index %d', 1, expStr, ...
            model.params.index(1));
        for ip= 2:nParam,
            expStr = '';
            if strcmp(model.params.scale{ip},'log'), expStr = '10^'; end
            fprintf(', %d: %sP -> index %d ', ip, expStr, ...
                model.params.index(ip));
        end
        fprintf('\n');
    end
end

% no Xvalidation, calculate errors
EE= zeros(nCombos,2);
SE= zeros(nCombos,2);
for ip= 1:nCombos,
    for iv= 1:nParam,
        % Build a concrete modell with a combination of parameters
        if strcmpi(model.params.scale(iv), 'log'),
            v= 10^model.params.param{ip}(iv);
        else
            v= model.params.param{ip}(iv);
        end
        modell.classy{model.params.index(iv)}= v;
    end
    parStr= sprintf('parcomb%d= [%s]> ', ip, ...
        vec2str(model.params.param{ip}, '%g'));
    if opt.fixed_sets & ~isempty(opt.er) & ~isempty(opt.er{ip})
        EE(ip,:)= opt.er{ip};
        SE(ip,:)= opt.variance{ip};
        if opt.verbosity,
            fprintf([parStr fmt opt.dsply_plusminus fmt ...
                '  (result from above)\n'], EE(ip,1), SE(ip,1));
        end
    else
        opt.out_prefix= parStr; %% otherwise possibly overwritten by progress bar
        [EE(ip,:), SE(ip,:), out_test]= feval(opt.estimate_ge, dat, modell.classy, opt);
    end
end

% choose optimal configuration (maybe there are some ties)
[dummy, best] = min(EE(:,1)+opt.std_factor*SE(:,1));
E = EE(best,1);
S = SE(best,1);
c = find( EE(:,1) <= E+S*opt.std_consider );
if opt.verbosity,
    parStr= cell(1, nParam);
    for ip= 1:nParam,
        expStr='';
        if strcmp(model.param(ip).scale,'log'), expStr = '10^'; end
        parStr{ip}= sprintf('%s%g', expStr, model.params.param{best}(ip));
    end
    optStr= '';
    if opt.std_factor>0,
        optStr= '(under consideration of the std) ';
    end
    fprintf(['minimum loss ' optStr 'was ' fmt ' at <%s>\n\n'], ...
        EE(best,1), vec2str(parStr,'%s','/'));
end

% get the datas belonging to the considered results
classy = model.classy;
for ip= 1:nParam,
    if strcmpi(model.param(ip).scale, 'log'),
        v= 10^model.params.param{best}(ip);
    else
        v= model.params.param{best}(ip);
    end
    classy{model.params.index(ip)}= v;
    for k= 1:length(c),
        PA{k}(ip) = model.params.param{c(k)}(ip);
    end
    P(ip) = model.params.param{best}(ip);
end
for k= 1:length(c),
    SA{k} = SE(c(k),:);
    ERRO{k} = EE(c(k),:);
end


% msDepth = 1 (or E=0) -> search has finished
if opt.msDepth == 1 | E == 0,
    ms_memo.model = model;
    if (exist('mesha'))
        ms_memo.mesha = mesha; % seems to work only for original search of msDepth==1 (without recursion), but fails for recursive model selection search! Michael, 2012_06_27
    end
    ms_memo.EE = EE;
    return;
end


%Depthsearch

%find interesting parameters

value= model.params.value;

%find all neighbors to the best parameters
for k= 1:length(c),
    para{k}{1} = PA{k};
    for i= 1:nParam,
        % Position of the best parameter in the value list
        pl = find(para{k}{1}(i) == model.params.value{i});
        % Size of the interval to the left neighbor:
        if pl>1,
            lef = model.params.value{i}(pl)-model.params.value{i}(pl-1);
        elseif model.params.value{i}(1)>model.params.minim(i)
            % Ooops, there is no left neighbor in the list: Take the interval
            % to the set minimum value
            lef = (model.params.value{i}(pl)-model.params.minim(i));
            if opt.depth==1,
                lef=unique([2*lef,lef,2*max(lef,model.params.value{i}(min(2,length(model.params.value{i})))-model.params.value{i}(1))]);
            end % to see for the corners
        else
            lef= [];
        end
        % Size of the interval to the right neighbor:
        if pl < length(model.params.value{i}),
            rig = model.params.value{i}(pl+1)-model.params.value{i}(pl);
        elseif model.params.value{i}(length(model.params.value{i})) ...
                <model.params.maxim(i)
            % There is no right neighbor: Take the interval to the maximum value
            rig = -(model.params.value{i}(pl)-model.params.maxim(i));
            if opt.depth==1,
                rig=unique([2*rig,rig,2*max(rig,model.params.value{i}(end)-model.params.value{i}(max(1,length(model.params.value{i})-1)))]);
            end % to see for the corners
        else
            rig= [];
        end
        abc = length(para{k});
        if ~isempty(lef),
            for j= 1:abc,
                for ll = 1:length(lef)
                    para{k}{length(para{k})+1} = para{k}{j};
                    para{k}{length(para{k})}(i) = para{k}{length(para{k})}(i)-0.5*lef(ll);
                    cc = find(para{k}{length(para{k})}(i) <=value{i}(:));
                    if isempty(cc),
                        value{i}(length(value{i})+1) = para{k}{length(para{k})}(i);
                    elseif ~(para{k}{length(para{k})}(i) == value{i}(min(cc))),
                        for m= length(value{i}):-1:(min(cc)),
                            value{i}(m+1) = value{i}(m);
                        end
                        value{i}(min(cc)) = para{k}{length(para{k})}(i);
                    end
                end
            end
        end
        if ~isempty(rig),
            for j= 1:abc,
                for ll = 1:length(rig)
                    para{k}{length(para{k})+1} = para{k}{j};
                    para{k}{length(para{k})}(i) = ...
                        para{k}{length(para{k})}(i)+0.5*rig(ll);
                    cc = find(para{k}{length(para{k})}(i) <= value{i}(:));
                    if isempty(cc),
                        value{i}(length(value{i})+1) = para{k}{length(para{k})}(i);
                    elseif ~(para{k}{length(para{k})}(i) == value{i}(min(cc))),
                        for m= length(value{i}):-1:min(cc),
                            value{i}(m+1) = value{i}(m);
                        end
                        value{i}(min(cc)) = para{k}{length(para{k})}(i);
                    end
                end
            end
        end
    end
end

if isfield(model,'integer'),
    for k= 1:length(c),
        for j= 1:length(para{k}),
            para{k}{j}(model.integer) = round(para{k}{j}(model.integer));
        end
    end
end

for k = 1:length(para)
    abcde = cat(1,para{k}{:});
    abcde = max(abcde,repmat(model.params.minim,[size(abcde,1),1]));
    abcde = min(abcde,repmat(model.params.maxim,[size(abcde,1),1]));
    para{k} = num2cell(abcde,2);
end

% mesh all neighbors
pa= 0;
for k= 1:length(c),
    for j= 1:length(para{k}),
        flag= 1;
        for i= 1:pa,
            if all(abs(param{i}-para{k}{j})<=eps),
                flag= 0;
                if j==1,
                    Er{i}= ERRO{k};
                    Se{i}= SA{k};
                end
            end
        end
        if flag,
            pa= pa+1;
            param{pa}= para{k}{j};
            if j==1,
                Er{pa}= ERRO{k};
                Se{pa}= SA{k};
            else
                Er{pa}= [];
                Se{pa}= [];
            end
        end
    end
end

% all neighbors found, now prepare the recursion
model.params.value = value;
model.params.param = param;
opt.er = Er;
opt.variance = Se;
opt.depth = opt.depth+1;
opt.msDepth = opt.msDepth-1;
[classy,E,S,P,out_test,ms_memo] = select_model(dat, model, opt);
