function opt = gaussProc_prepareKernels(Xtrain, opt, prepareParams)
% gaussProc_prepareKernels - GP helper function: Consistency checks for kernels
%
% Synopsis:
%   opt = gaussProc_prepareKernels(opt,prepareParams)

[dim, N] = size(Xtrain);
% Prepare for multiple kernel learning: If only one kernel function is
% given, convert it to a cell array. We want to have all cases unified
% such that each kernel is specified by one entry in opt.kernel
if ~iscell(opt.kernel),
  % Kernel given by function name (string or handle)
  opt.kernel = {{opt.kernel}};
  if opt.verbosity>1,
    fprintf('Option ''kernel'': You are using 1 kernel.\n');
  end
else
  % Kernel is given in cell: If all entries are again cell arrays, this
  % is a valid multiple kernel cell
  onlyCells = 1;
  for i = 1:length(opt.kernel),
    onlyCells = onlyCells & iscell(opt.kernel{i});
  end
  if onlyCells,
    if opt.verbosity>1,
      fprintf('Option ''kernel'': You are using a linear combination of %i kernels.\n', ...
              length(opt.kernel));
    end
  else
    % Kernel is given as cell, but contents is not another cell: We have
    % only one kernel (eg. {'rbf', 'bias', 1}), pack into another cell array
    opt.kernel = {{opt.kernel}};
    if opt.verbosity>1,
      fprintf('Option ''kernel'': You are using 1 kernel.\n');
    end
  end
end

nKernels = length(opt.kernel);
% For each kernel, we have a positive scalar determining the
% weight. Check whether kernelweights option has suitable size
if isempty(opt.kernelweight),
  % Nothing given: Assume equal weights for each kernel, weights sum to 1
  opt.kernelweight = log(1/nKernels)*ones([1 nKernels]);
elseif length(opt.kernelweight)~=nKernels,
  error('Length of option ''kernelweights'' musst match number of kernels');
end
% Allow data indexing via the kernelindex parameter: Each index array
% corresponds to one kernel
if ~isempty(opt.kernelindex) & length(opt.kernelindex)~=nKernels,
  error('Length of option ''kernelindex'' musst match number of kernels');
end
% Just some info for the user:
if opt.verbosity>1,
  for i = 1:nKernels,
    if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
      fprintf('Option ''kernelindex'': Kernel %i operates on all %i features.\n', ...
              i, dim);
    else
      fprintf('Option ''kernelindex'': Kernel %i operates on a subset of %i features.\n', ...
              i, length(opt.kernelindex{i}));
    end
  end
end

if prepareParams,
  for i = 1:nKernels,
    % Prepare for optimization. First, we need to extract the list of all
    % kernel parameters for gradient computation
    [kernelFunc, kernelParam] = getFuncParam(opt.kernel{i});
    % The kernel derivative routines need to return a list of all kernel
    % parameters. Also, pass kernel parameters to this function, so that we
    % can use the expanded options structure later (second return arg)
    if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
      % No index given at all or nothing given for this particular
      % kernel: assume they operate on all data
      [allParams, kernelOpt] = ...
          feval(['kernderiv_' kernelFunc], Xtrain, [], 'returnParams', kernelParam{:});
    else
      % Index given:
      [allParams, kernelOpt] = ...
          feval(['kernderiv_' kernelFunc], Xtrain(opt.kernelindex{i},:), [], ...
                'returnParams', kernelParam{:});
    end
    kernelOpt.allParams = allParams;
    % Replace the old entry for the kernel by the fully expanded options
    opt.kernel{i} = {kernelFunc, kernelOpt};
  end
end
