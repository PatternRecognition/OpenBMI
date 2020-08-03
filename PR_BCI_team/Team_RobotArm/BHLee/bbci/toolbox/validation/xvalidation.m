function [outarg1, loss_std, out_test, memo]= xvalidation(epo, model, varargin)
%[loss, loss_std, out_test, memo]= xvalidation(fv, model, <opt>)
%
% IN  fv      - features vectors, struct with fields
%               .x (data, nd array where last dim is nSamples) and
%               .y (labels [nClasses x nSamples], 1 means membership)
%     model   - name or structure specifying a classification model,
%               cd to /classifiers and type 'help Contents' to see a list.
%               if the model has free parameters, they are chosen by
%               select_model on each training set (default, time consuming),
%               or by calling select_model beforehand, see opt.outer_ms
%               (quicker, but can bias the estimation of xvalidation loss).
%     opt:
%     .sample_fcn   - name of the function that is called to randomly
%                     draw training / test set splits (prefix 'sample_' is
%                     automatically prepended). The dataset is usually
%                     partitioned into #folds equisized folds, with 1 fold being 
%                     used for validation(test) and the other #folds-1 for training. 
%                     This is usually repeated #folds times, so that each
%                     single fold served for validation. If there are
%                     multiple shuffles, this procedure is repeated
%                     #shuffles times.
%                     Examples (default 'divisions'): 
%                     'divisions'      - in each fold, between-class ratios (ie, the relative number of samples per class) are preserved 
%                     'kfold'          - random partitioning of the samples into folds, class memberships are ignored
%                     'chronKfold'     - partitioning in chronological order instead of random. Multiple #shuffles are not possible and will be ignored.
%                     'chronKKfold'    - ?
%                     'chronSplit'     - ?
%                     'leaveOneOut'    - validation on single sample, training on rest. This is repeated for each sample.
%                     'evenOdd'        - split training samples alternately. Esp. for small training sets: used for checking, if single outliers produce a bias.
%                     'fixedTrainsize' - you can specify a fixed size for the training set (for each class). Validation is performed on the remaining data.
%                     'divisionsEqui'  - ??
%                     'divisionsPlus'  - ??
%                     Can also be a cell array, where the first cell holds
%                     the name of the function as string, and the following
%                     cells hold parameters that are passed to the
%                     sample_fcn.
%     .xTrials      - <'xTrials', xt> is a short cut for
%                     <'sample_fcn', {'divisions', xt}>. The value xt is a
%                     vector [#shuffles, #folds], see sample_divisions,
%                     default [10 10].
%     .loss         - string (loss_fcn), or cell array {loss_fcn, loss_param},
%                     e.g., '0_1' (default), 'classwiseNormalized', 'identity',
%                     {'byMatrix', loss_matrix}
%                     if first choice would call the function loss_0_1 to
%                     determine the loss, the last choice would call
%                     loss_byMatrix and pass the argument loss_matrix.
%                     The loss function 'identity' can be used to
%                     evaluate likelihoods, see LOSS_IDENTITY. For this
%                     loss function, EPO.y is effectively ignored and
%                     thus need not be provided.
%     .ms_sample_fcn- like sample_fcn but to be used in model selection.
%     .msTrials     - like xTrials, but for model selection. when the third
%                     value is -1 it is set to the size of the training set
%                     (useful for outer model selection), default [3 10]
%                     without .outer_ms and [3 10 -1] with .outer_ms.
%     .outer_ms     - perform model selection before the xvalidation.
%                     this can bias the estimation of the xvalidation loss.
%                     should not be done on the whole set, e.g., by
%                     choosing opt.msTrials= [3 10 -1]. default 0.
%     .divTr,.divTe - specified fixed training / test set splits
%                     format must be as the output of the sample_* functions.
%                     if this field is given .sample_fcn is (of course)
%                     ignored.
%                     alternatively these fields can be added to the fv
%                     structure (first argument)
%     .msdivTr, .msdivTe - specified fixed training / test set splits for
%                     outer model selection (only possible is .outer_ms=1),
%                     otherwise the same as above.
%     .proc         - class dependent preprocessing; this string is
%                     evaluated for each training set (labels of the test
%                     set are set to NaN),
%                     the procedure must calculate 'fv' from 'fv'.
%                     Obsolete is to use a procedure which calculates
%                     'fv' from 'epo'. This still works, but maybe will
%                     be restricted in future versions.
%                     (field proc can also be given in fv structure)
%                     EXTENDED to free parameters, to be explained.
%     .out_trainloss- output not only test but also training loss. in this
%                     case loss and loss_std are 2-d vectors, default 0.
%     .std_of_means - if true output loss_std is calculated as std of the
%                     means of the losses of each fold, default 1.
%     .classifier_nargout - specify the number of output-arguments to
%                     receive from the applyFcn. If this is set > 1 then
%                     out_test is made a structure with a field for each
%                     argout and all nargouts are passed to the loss_fcn 
%                     (which needs to be able the handle them). All argouts
%                     are assumed to be row-vectors with one number for
%                     each test point, like the usual continous classifier-out.
%     .catch_trainerrors - if true, errors when calling the training
%                     routine do not cause xvalidation to stop. Instead,
%                     the respective predictions will be set of NaN. Mind
%                     that errors during model selection will still cause
%                     xvalidation to abort. default 0.
%     .allow_reject - if true, classifier outputs that are NaN will be
%                     interpreted as "rejected" (can not be classified
%                     with minimum level of certainty). See below for
%                     changes in the output of xvalidation. default: 0
%     .allow_transduction - if true, also the samples of the test set are
%                     passed to the classifier, but (of course) with NaN'ed
%                     labels, default 0.
%     .verbosity    - level of verbosity (0: silent, 1: normal,
%                     2: show intermediate results of model selection)
%     .save_classifier - saves the classifier that was selected by an
%                     inner model selection into the memo output variable,
%                     default 0.
%     .save_proc    - saves the processing that was selected by an inner
%                     process selection into the memo output variable.
%     .save_proc_params - a cell array of variable names that are free
%                     variables or output variables of opt.proc and that
%                     were selected by an inner process selection into
%                     the memo output variable.
%     .save_file    - a name of a file intermediate results will be saved, too.
%                     If this fields is defined, a restart of this programm
%                     will start at the point the last intermediate result
%                     was saved. This is important for condor.
%     .clock        - show a work progress clock (in figure)
%     .progress_bar - show a work progress bar (in terminal)
%     .out_timing   - prints out elapsed time, default 1.
%     .out_prefix   - a string that is printed before the result
%                     (unfinished lines printed before calling xvalidation
%                     are possibly erased by the progress bar)
%     .train_only   - return (as first output argument instead of loss)
%                     trained classifiers (in a cell array) for each fold
%                     of the  cross-validation. classifiers are not
%                     applied to the test sets.
%     .train_jits   - for training use only samples with those jitter
%                     indices (requires that fv has field jit)
%     .test_jits    - for testing use only samples with those jitter
%                     indices (requires that fv has field jit)
%     .fp_bound     - <do we still need this?>
%
%     .twoClass2Bool - if true (default) and nClasses == 2 xvalidation will 
%                      detect the binary problem and change fv.y to a
%                      vector [1 x nSamples] with 0/1.
%
% OUT loss      - loss of the test set
%                 if opt.out_trainloss is true, [loss_test, loss_train]
%     loss_std  - analog, see opt.std_of_means
%     out_test  - continuous classifier output for each x-val trial /
%                 sample, a matrix of size [nClasses nSamples nTrials].
%                 For classifiers with reject option: see below.
%     memo      - according to opt.save_classifier, opt.save_proc, and
%                 opt.save_proc_params saves some of the selections that
%                 are made by a model/process selection within the
%                 cross-validation.
%
% You can make 'equivalence classes' of samples by adding a field 'bidx'
% (= 'base index') to the fv. This must be a vector of length #samples
% specifying an index of equivalence class. This has the effect that
% all samples that belong to one equivalence classes (i.e., they have
% the same index) are either *all* put into the training set or
% *all* put into the test set. In this case indices in the training /
% test set splits refer to these indices of equivalence classes.
%
% To make further evaluations of the output, you can call functions
% from the family val_*, e.g,
%   val_rocCurve to plot a ROC curve,
%   val_confusionMatrix to calculate the confusion matrix,
%   val_compareClassifiers to see whether your result is 'significantly'
%                          better than chance, ...
%
% If you want to reject outliers, you have to implement a function
% proc_outl_* that has as input and output a fv structure, which sets
% the labels (field .y) of rejected trials to 0 or NaN. Then set
% opt.proc to 'fv= proc_outl_*(fv, <possible_some_parameters>);'.
% Caveat: the loss on the training set (which can display by setting
% opt.out_trainloss to 1) is then calculated on the *accepted trials only*.
% Future versions of xvalidation may allow also to reject samples to
% the test sets.
%
% XVALIDATION with the allow_reject option handles classifiers that can
% reject samples in the apply phase (for example, if an example can not be
% classified with a minimum certainty). Rejected samples need to be marked
% as NaNs in the classifier output. With this option set, output variable
% out_test is a structure with 2 fields:
%   'out': the continuous classifier output as a matrix of size
%     [nClasses nSamples nTrials] (or [1 nSamples nTrials] for 2 classes)
%   'valid': a logical array of size [nSamples nTrials].
%     out_test.valid(i,j)==1 means that example i has not been rejected in the
%     j.th cross-validation trial.
% Mind that some of the samples may not have been used as test
% examples. Such examples are also marked by NaNs in out_test, but their
% corresponding out_test.valid entry is 1.
% Rejected samples are passed to the loss function without any
% modification. Make sure that the loss function handles rejected samples
% in a meaningful way.
%
%
% SEE select_model, select_proc, sample_kfold, sample_divisions, loss_0_1,loss_identity

% bb, ida.first.fhg.de
% with extensions by guido and anton
%
% Copyright Fraunhofer FIRST.IDA (2004)
% $Id$

t0= cputime;

if length(varargin)==1 && isreal(varargin{1}),
  opt.xTrials= varargin{1};
else
  opt= propertylist2struct(varargin{:});
end

if isfield(epo, 'y'),
  % Standard case with given labels:
  labelsProvided = 1;
  [nClasses, nSamples]= size(epo.y);
  % Class labels could also be logicals. Nice in principle, but later we
  % will set some class labels to NaN, and this is not possible with
  % logicals.
  if islogical(epo.y),
    epo.y = double(epo.y);
  end
else
  % No labels given: Assume that the task is to evaluate likelihoods. Create a
  % dummy label field, so that all subsequent code works without change. To
  % make the various sampling procedures work nicely, I create a dummy field
  % indicating that everything is from one class
  labelsProvided = 0;
  nSamples = size(epo.x, ndims(epo.x));
  nClasses = 1;
  epo.y = ones([nClasses nSamples]);
end

[opt, isdefault]= set_defaults(opt, ...
                               'xTrials', [10 10], ...
                               'loss', '0_1', ...
                               'sample_fcn', 'divisions', ...
                               'outer_ms', 0, ...
                               'msTrials', [3 10], ...
                               'ms_sample_fcn', 'divisions', ...
                               'out_trainloss', 0, ...
                               'out_timing', 0, ...
                               'out_sampling', 0, ...
                               'out_prefix', '', ...
                               'std_of_means', 1, ...
                               'classifier_nargout', 1, ...
                               'catch_trainerrors', 0, ...
                               'allow_reject', 0, ...
                               'train_only', 0, ...
                               'block_markers',[],...
                               'proc', '', ...
                               'fp_bound', 0, ...
                               'allow_transduction', 0, ...
                               'verbosity', 1, ...
                               'clock', 0, ...
                               'save_file','',...
                               'progress_bar', 1, ...
                               'divTr', [], ...
                               'msdivTr', [], ...
                               'save_classifier', 0, ...
                               'save_proc', 0, ...
                               'save_proc_params', [], ...
                               'debug', 0, ...
                               'dsply_precision', 3, ...
                               'dsply_plusminus', char(177),...
                               'twoClass2Bool',true);
%                 'dsply_plusminus', '+/-');

% In the implementation, the progress_bar option overrides
% verbosity. Fix this, so that nothing is displayed with verbosity==0 and
% progress_bar unchanged from its default
if opt.verbosity<1 && isdefault.progress_bar,
  opt.progress_bar = 0;
end
if isequal(opt.ms_sample_fcn, 'divisions'),
  if opt.outer_ms && isdefault.msTrials,
    if isfield(opt, 'xTrials'),
      opt.msTrials= [opt.xTrials(1:2) -1];
    else
      opt.msTrials= [3 10 -1];
    end
  end
  opt.ms_sample_fcn= {opt.ms_sample_fcn, opt.msTrials};
else
  if ~isdefault.msTrials,
    msg= 'property .msTrials is ignored when you specify .ms_sample_fcn';
    bbci_warning(msg, 'validation', mfilename);
  end
end
if isequal(opt.sample_fcn, 'divisions'),
  opt.sample_fcn= {opt.sample_fcn, opt.xTrials};
  opt= rmfield(opt, 'xTrials');
else
  if ~isdefault.xTrials,
    msg= 'property .xTrials is ignored when you specify .sample_fcn';
    bbci_warning(msg, 'validation', mfilename);
  end
end
opt= rmfield(opt, 'msTrials');

if isfield(epo, 'proc'),
  if ~isempty(opt.proc),
    error('field proc must either be given in fv or opt argument');
  end
  opt.proc= epo.proc;
  epo= rmfield(epo, 'proc');
end

if ~isempty(opt.proc) && isstruct(opt.proc),
  if isfield(opt.proc, 'eval'),
    if isfield(opt.proc, 'train') || isfield(opt.proc, 'apply'),
      error('proc may either have field eval XOR fields .train/.apply');
    end
  else
    if ~isfield(opt.proc, 'train'),
      opt.proc.train= '';
    end
    if ~isfield(opt.proc, 'apply'),
      opt.proc.apply= '';
    end
  end
end
if opt.debug,
  if isdefault.progress_bar,
    opt.progress_bar= 0;
  end
  persistent counter
  if opt.debug==1,
    counter= 1;
  else
    counter= counter+1;
  end
  fprintf('xv (#%d, depth %d), %s', counter, opt.debug, ...
          toString(opt.sample_fcn));
  if prochasfreevar(opt.proc),
    fprintf(', ');
    if opt.outer_ms,
      fprintf('outer');
    end
    fprintf('proc selection');
  elseif isfield(opt.proc, 'pvi'),
    fprintf(', proc indices %s', vec2str(opt.proc.pvi));
  end
  if isstruct(model),
    fprintf(', ');
    if opt.outer_ms,
      fprintf('outer');
    end
    fprintf('model selection');
  end
  fprintf('\n');
  opt.debug= opt.debug+1;
end

if opt.save_proc && ~prochasfreevar(opt.proc),
  msg= 'save_proc makes only sense for .proc with free variables';
  bbci_warning(msg, 'validation', mfilename);
  opt.save_proc= 0;
end

if isfield(epo, 'divTr'),
  if ~isempty(opt.divTr),
    error('divTr is given in both, fv and opt argument');
  end
  opt.divTr= epo.divTr;
  opt.divTe= epo.divTe;
  epo= rmfield(epo, {'divTr','divTe'});
end

if isfield(epo, 'msdivTr'),
  if ~isempty(opt.msdivTr),
    error('msdivTr is given in both fv and opt argument');
  end
  opt.msdivTr= epo.msdivTr;
  opt.msdivTe= epo.msdivTe;
  epo= rmfield(epo, {'msdivTr','msdivTe'});
end
if ~isempty(opt.msdivTr) && ~opt.outer_ms,
  error('msdivTr can only be specified for outer model selection');
end

fmt= ['%.' int2str(opt.dsply_precision) 'f'];

opt= set_defaults(opt, 'check_bidx_labels',~isfield(opt,'divTr'));

if isfield(epo, 'bidx'),
  [repIdx, eqcl]= xval_choose_repIdx(epo, opt.check_bidx_labels);
else
  % "Normal" data without bidx: Need to distinguish between regression
  % and classification here. For regression, epo.y might take on the
  % value zero, kicking that effectively sample out 
  if size(epo.y,1)==1 && length(unique(epo.y))>2,
    isRegression = 1;
    repIdx = 1:length(epo.y);
  else
    isRegression = 0;
    repIdx= find(any(epo.y,1));
  end
  % bidx binds samples to equivalence classes.
  % Non-valid samples are bound to class "0", which is never sampled.
  epo.bidx = zeros(1, size(epo.y,2));
  epo.bidx(repIdx) = repIdx;
  eqcl = repIdx;
end
if ~isfield(epo, 'jit'),
  epo.jit= zeros(size(epo.bidx));
end

opt= set_defaults(opt, ...
                  'train_jits', unique(epo.jit), ...
                  'test_jits', unique(epo.jit));


save_interm_vars = {};
opt_ms= [];
%if ~isstruct(model),   %% classifier without free hyper parameters
%  classy= model;
%  model= [];
%end
if prochasfreevar(opt.proc) || isstruct(model),
  %% either selection of pre-processing parameter or selection
  %% of classification model is required
  opt_ms= copy_struct(opt, 'not', 'save_file','save_proc','save_classifier',...
                      'train_only');
  opt_ms.clock= 0;
  opt_ms.verbosity= max(0, opt.verbosity-1);
  opt_ms.sample_fcn= opt.ms_sample_fcn;
  opt_ms.divTr= opt.msdivTr;
  if isfield(opt, 'msdivTe'),
    opt_ms.divTe= opt.msdivTe;
  end
  if opt.outer_ms,
    loaded = 0;
    if ~isempty(opt.save_file) && exist([opt.save_file,'.mat'],'file')
      S = load(opt.save_file);
      if isfield(S,'classy')
        load(opt.save_file);
        loaded = 1;
      end
    end
    if loaded==0
      if opt.verbosity>0,
        msg= 'outer model selection can bias the results';
        bbci_warning(msg, 'validation', mfilename);
      end
      if opt.verbosity<2,
        opt_ms.progress_bar= 0;
      end
      opt_proc_memo= opt.proc;
      %      if isstruct(opt.proc) & isfield(opt.proc, 'train'),
      %        proc= copy_struct(opt.proc, 'not','train','apply');
      %        proc.eval= opt.proc.train;
      %        [proc, classy, ml, mls]= select_proc(epo, model, proc, opt_ms);
      %        if isfield(proc, 'param'),
      %          opt.proc.param= proc.param;
      %        end
      %      else
      [opt.proc, classy, ml, mls]= select_proc(epo, model, opt.proc, opt_ms);
      %      end
      if ~isequal(opt.proc, opt_proc_memo) && opt.verbosity>0,
        fprintf('select_proc chose: ');
        disp_proc(opt.proc);
      end
      if isstruct(model),
        if opt.verbosity>0,
          fprintf(['chosen classifier: %s -> ' ...
                   fmt opt.dsply_plusminus fmt '\n'], ...
                  toString(classy), ml(1), mls(1));
        end
      end
      model= [];  %% model has been chosen -> classy
      if ~isempty(opt.save_file)
        save_interm_vars = {'opt_proc_memo','classy','ml','mls','opt','model'};
        save(opt.save_file,save_interm_vars{:},'save_interm_vars');
      end
    end
  else  %% i.e., not opt.outer_ms
    opt_ms.verbosity= 0;
    opt_ms.progress_bar= 0;
    if isstruct(model),
      classy= model.classy;  %% This is defined only to get the names,
                             %% the parameters have to determined on the
                             %% training sets within the cross-validation.
    else
      classy= model;
      model= [];
    end
  end
else              %% classification model without free hyper parameters
  classy= model;
  model= [];
end

[func, train_par]= getFuncParam(classy);
train_fcn= ['train_' func];
applyFcn= getApplyFuncName(classy);

[func, loss_par]= getFuncParam(opt.loss);
loss_fcn= ['loss_' func];

% Issue a warning if 'loss_identity' is used with labels given, as the
% labels will be ignored in this case
if strcmp(loss_fcn, 'loss_identity') && labelsProvided,
  warning('With loss function LOSS_IDENTITY, labels EPO.y will be ignored.');
end

if opt.fp_bound~=0 && ~isequal(applyFcn, 'apply_separatingHyperplane'),
  error('FP-bound works only for separating hyperplane classifiers');
end

if ~isempty(opt.divTr),
  divTr= opt.divTr;
  %% special feature: when divTr is given, but not divTe: take as test
  %% samples all those, which are not in the training set.
  if ~isfield(opt, 'divTe') && ~opt.train_only,
    opt.divTe= cell(1,length(divTr));
    for nn= 1:length(divTr),
      opt.divTe{nn}= cell(1,length(divTr{nn}));
      for kk= 1:length(divTr{nn}),
        opt.divTe{nn}{kk}= setdiff(1:max(eqcl), opt.divTr{nn}{kk});
      end
    end
  end
  divTe= opt.divTe;
  for i = 1:length(opt.divTr)
    for j = 1:length(opt.divTr{i})
      for k = 1:length(opt.divTr{i}{j})
        divTr{i}{j}(k) = find(eqcl==opt.divTr{i}{j}(k));
      end
      if ~opt.train_only
        for k = 1:length(opt.divTe{i}{j})
          divTe{i}{j}(k) = find(eqcl==opt.divTe{i}{j}(k));
        end
      else
        divTe{i}{j} = [];
      end
    end
  end
  sample_fcn= 'given sample partitions';
  sample_params= {};
else
  [func, sample_params]= getFuncParam(opt.sample_fcn);
  sample_fcn= ['sample_' func];
  [divTr, divTe]= ...
      feval(sample_fcn, epo.y(:,repIdx), sample_params{:});
end
check_sampling(divTr, divTe);

if isfield(epo, 'equi') || ...
        (isfield(opt, 'equi') && ~strcmp(sample_fcn, 'divisionsEqui')),
    error(['field equi must now be passed in opt structure within the field ' ...
        'sample_params and sample_fcn must be divisionsEqui']);
end

label= epo.y;

%#########################################
% exception for binary classification tasks
%
%  changed by stl at 18.02.2005
%

if (opt.classifier_nargout > 1),
  %%% avoid calling loss_fcn b.c. classifier output is not yet available
  %%%
  %%% does 1 or 0 make more sense here?
  loss_samplewise= 1;
else
  if size(label) == 2, % shouldn't this rather be "size(label,1) == 2" ?? (Michael, 2012_07_10)
    l= feval(loss_fcn, label, label(1,:), loss_par{:});
  else
    l= feval(loss_fcn, label, epo.y, loss_par{:});
  end ;
  %#########################################
  loss_samplewise= (length(l)>1);
end

if ~loss_samplewise,
  if opt.out_trainloss,
    msg= sprintf('trainloss cannot be returned for loss <%s>', loss_fcn);
    bbci_warning(msg, 'validation', mfilename);
    opt.out_trainloss= 0;
  end
end

nTrials= length(divTe); % for better understanding, consider renaming "nTrials" with "nShuffles"
if ~opt.train_only,
    avErr= NaN*zeros(nTrials, length(divTe{1}), opt.out_trainloss+1);
    if opt.twoClass2Bool
      out_test= NaN*zeros([nClasses-(nClasses==2), nSamples, nTrials]);
    else
      out_test= NaN*zeros([nClasses, nSamples, nTrials]);      
    end
    % Store for each sample whether the classifier has produced a valid
    % output, i.e., it has not rejected the sample.
    out_valid = logical(ones([nSamples nTrials]));

    % Make a cell array for all the classifier output apart from the
    % first one (the first output arg is handled by the old xvalidation
    % code, don't want to touch that)
    more_out_test = cell([1 opt.classifier_nargout-1]);
    for tmp = 1:length(more_out_test),
      if opt.twoClass2Bool
        more_out_test{tmp} = NaN*zeros([nClasses-(nClasses==2), nSamples, nTrials]);
      else
        more_out_test{tmp} = NaN*zeros([nClasses, nSamples, nTrials]);        
      end
    end
end

memo= [];

if ~isempty(opt.save_file) & exist([opt.save_file,'.mat'],'file')
    load(opt.save_file);
    if ~exist('n','var')
        n0 = 1;
        d0 = 1;
    else
        n0 = n;
        d0 = d+1;
    end
else
    n0 = 1;
    d0 = 1;
end

if ~isfield(epo,'classifier_param')
    epo.classifier_param = {};
end

if opt.progress_bar, tic; end
for n= n0:nTrials,
    nDiv= length(divTe{n});  %% might differ from nDivisions in 'loo' case
    usedForTesting = logical(zeros([1 size(out_test,2)]));
    for d= d0:nDiv,
        if opt.debug==2 && (isstruct(model) || prochasfreevar(opt.proc)),
            fprintf('xv: division [%d %d]\n', n, d);
        end
        k= d+(n-1)*nDiv;
        bidxTr= divTr{n}{d};
        bidxTe= divTe{n}{d};
        idxTr= find(ismember(epo.bidx, epo.bidx(repIdx(bidxTr))) & ...
            ismember(epo.jit, opt.train_jits));
        idxTe= find(ismember(epo.bidx, epo.bidx(repIdx(bidxTe))) & ...
            ismember(epo.jit, opt.test_jits));
        epo.y(:,idxTe)= NaN;              %% hide labels of the test set

        if ~isempty(model),               %% do model selection on training set
            fv= xval_selectSamples(epo, idxTr);
            [best_proc, classy, E, V,ms_memo]= select_proc(fv, model, opt.proc, opt_ms);
            memo.ms_memo{n}{d} = ms_memo; % keep memo from model selection for post-observations
            if prochasfreevar(best_proc),
                error('not all free variable were bound');
            end
            [func, train_par]= getFuncParam(classy);
        elseif prochasfreevar(opt.proc),
            fv= xval_selectSamples(epo, idxTr);
            %      if isstruct(opt.proc) & isfield(opt.proc, 'train'),
            %        proc= copy_struct(opt.proc, 'not', 'train','apply');
            %        proc.eval= opt.proc.train;
            %      else
            %        proc= opt.proc;
            %      end
            best_proc= select_proc(fv, classy, opt.proc, opt_ms);
            if prochasfreevar(best_proc),
                error('not all free variable were bound');
            end
        else
            best_proc= opt.proc;
        end
        % Select the data points for train and test for the current fold, and separately apply the preprocessing functions
        idxTrTe= [idxTr, idxTe];
        iTr= 1:length(idxTr);
        iTe= length(idxTr) + [1:length(idxTe)];
        if isstruct(best_proc) && isfield(best_proc, 'train'),
            fv1= xval_selectSamples(epo, idxTr);
            proc= copy_struct(best_proc, 'not', 'train','apply');
            proc.eval= best_proc.train;
            [fv1, proc]= proc_applyProc(fv1, proc);
            proc.eval= best_proc.apply;
            fv2= xval_selectSamples(epo, idxTe);
            fv2= proc_applyProc(fv2, proc);
            fv= proc_appendSamples(fv1, fv2);
            clear fv1 fv2;
            if isfield(proc, 'param'),
              best_proc.param= proc.param;
            end
        else
            fv= xval_selectSamples(epo, idxTrTe);
            fv= proc_applyProc(fv, best_proc);
        end
        fv= proc_flaten(fv);
        if size(fv.x,2)~=length(idxTrTe),
            error('number of samples was changed thru opt.proc!');
        end

        %% TODO: allow rejection of test samples ( -> performance criterium!)
        %    iRejectTe= find(any(isnan(fv.y(:,iTe))));
        iRejectTr= find(~any(fv.y(:,iTr)) | any(isnan(fv.y(:,iTr))));
        iTr(iRejectTr)= [];

        if opt.allow_transduction,
          %% pass also samples of the test set to the classifier
          %% (but with NaN'ed labels, of course)
          ff= xval_selectSamples(fv, [iTr iTe]);
        else
          ff= xval_selectSamples(fv, iTr);
        end
        if opt.catch_trainerrors,
          try
            C= feval(train_fcn, ff.x, ff.y, ff.classifier_param{:},train_par{:});
          catch
            if opt.verbosity>0,
              fprintf('Failed to train classifier in division [%d %d]\n', n, d);
            end
            C = [];
          end
        else
          C= feval(train_fcn, ff.x, ff.y, ff.classifier_param{:},train_par{:});
        end
        epo.y= label;

        if opt.fp_bound && ~isempty(C),
            iTeNeg=  iTe(find(epo.y(1,idxTe)));
            frac= floor(length(iTeNeg)*fp_bound);
            xp= C.w'*fv.x(:,iTeNeg);
            [so,si]= sort(-xp);
            C.b= so(frac+1) - eps;
        end
        if nargout>3,
            if opt.save_classifier,
                memo(k).C= C;
                memo(k).classy= classy;
            end
            if opt.save_proc,
                memo(k).proc= best_proc;
            elseif ~isempty(opt.save_proc_params),
                for fi= 1:length(opt.save_proc_params),
                    fld=  opt.save_proc_params{fi};
                    ip= strmatch(fld, {best_proc.param.var},'exact');
                    if isempty(ip),
                        msg= sprintf('variable %s not found', fld);
                        error(msg);
                    end
                    memo= setfield(memo, {k}, fld, ...
                        best_proc.param(ip).value{1});
                end
            end
        end
        if opt.train_only,
            outarg1(k)= C;
            if ~isempty(opt.save_file)
                save(opt.save_file,save_interm_vars{:},'n','d','divTr','divTe', ...
                    'memo','outarg1','save_interm_vars');
            end

        elseif ~isempty(C),
          % Outputs (out_test) already have NaN as default values. If training failed,
          % we don't need to do anything
          
          % A cell array to catch potential extra classfier outputs
          more_out = cell([1 opt.classifier_nargout-1]);
          if opt.out_trainloss,
            [out more_out{:}]= feval(applyFcn, C, fv.x);
            out_test(:, idxTe, n)= out(:,iTe);
            usedForTesting(idxTe) = 1;
            if (opt.classifier_nargout > 1),
              for tmp_num = 1:(opt.classifier_nargout-1),
                more_out_test{tmp_num}(:, idxTe, n) = more_out{tmp_num}(:,iTe);
              end
            end
            if loss_samplewise,
              loss= feval(loss_fcn, label(:,idxTrTe), out, more_out{:}, loss_par{:});
              avErr(n,d,1)= mean(loss(iTe));
              avErr(n,d,2)= mean(loss(iTr));
            else
              avErr(n,d,2)= feval(loss_fcn, label(:,idxTr), out(iTr), ...
                                  loss_par{:});
            end
          else
            [out more_out{:}]= feval(applyFcn, C, fv.x(:,iTe));
            out_test(:, idxTe, n)= out;
            usedForTesting(idxTe) = 1;
            if (opt.classifier_nargout > 1),
              for tmp_num = 1:(opt.classifier_nargout-1),
                more_out_test{tmp_num}(:, idxTe, n)= more_out{tmp_num};
              end
            end
            if loss_samplewise,
              loss= feval(loss_fcn, label(:,idxTe), out, more_out{:}, loss_par{:});
              avErr(n,d,1)= mean(loss);
            end
          end
          if opt.allow_reject,
            % Check for rejected samples: All classifier outputs need to be
            % NaN for rejected samples
            rejected = all(isnan(out), 1);
            out_valid(idxTe(rejected), n) = 0;
          end
          if ~isempty(opt.save_file)
            save(opt.save_file,save_interm_vars{:},'n','d','divTr','divTe',...
                 'avErr','out_test', 'more_out_test', 'memo','out_valid','save_interm_vars');
          end
        end
        if opt.progress_bar, print_progress(k, nDiv*nTrials); end
        if opt.clock, showClock(k, nDiv*nTrials); end
    end %% for d
    d0 = 1;
    if ~loss_samplewise,
      more_out_test_subset = {};
      % Subset all extra output arguments to those examples that were
      % ever used for testing
      for i_out = 1:length(more_out_test),
        more_out_test_subset = more_out_test{i_out}(:,usedForTesting,n);
      end
      avErr(n,:,1)= feval(loss_fcn, label(:,usedForTesting), ...
                          out_test(:,usedForTesting,n), ...
                          more_out_test_subset{:}, loss_par{:});
    end
end %% for n
if opt.train_only,
    return;
end

et= cputime-t0;
avE = mean(avErr,2);
avErr = reshape(avErr,[nTrials*length(divTe{1}), opt.out_trainloss+1]);
loss_mean= mean(avErr, 1);

outarg1= loss_mean;
if opt.std_of_means && loss_samplewise,
    loss_std= transpose(squeeze(std(avE, 0, 1)));
else
    loss_std= std(avErr, 0, 1);
end

% Make up the output for the case that some examples have been rejected
% by the classifier or more output arguments are to be taken care of
if opt.allow_reject || (opt.classifier_nargout > 1),
  out_test = struct('out', out_test);
  if opt.allow_reject,
    out_test.valid = out_valid;
  end
  if (opt.classifier_nargout > 1),
    out_test.more_out = more_out_test;
  end
end

if nargout==0 || opt.verbosity>0,
    if opt.out_sampling,
        if strcmp(sample_fcn, 'given sample partitions'),
            smplStr= sample_fcn;
        else
            smplStr= toString(sample_params);
            smplStr= sprintf('%s: %s', sample_fcn, smplStr(2:end-1));
        end
    end
    if opt.out_timing,
        timeStr= sprintf('%.1fs', et);
    end
    if opt.out_timing && opt.out_sampling,
        infoStr= sprintf('  (%s for %s)', timeStr, smplStr);
    elseif opt.out_timing,
        infoStr= sprintf('  (%s)', timeStr);
    elseif opt.out_sampling,
        infoStr= sprintf('  (on %s)', smplStr);
    else
        infoStr= '';
    end
    if opt.out_trainloss,
        fprintf([opt.out_prefix fmt opt.dsply_plusminus fmt ...
            ', [train: ' fmt opt.dsply_plusminus fmt ']' ...
            infoStr '\n'], [loss_mean; loss_std]);
    else
        fprintf([opt.out_prefix fmt opt.dsply_plusminus fmt infoStr '\n'], ...
            loss_mean, loss_std);
    end
end
