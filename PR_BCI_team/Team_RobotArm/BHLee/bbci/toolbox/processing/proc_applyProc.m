function [fv, SAVEVARproc]= proc_applyProc(fv, proc, pvi)
%[fv, proc]= proc_applyProc(fv, proc, <pvi>)
%
% Applies the fv transform defined in proc to the feature vectors fv. proc
% can either be a string to be evaluated with Matlabs EVAL function, or a
% struct array containing both a command string and values to set for
% parameter variables in the command string. When using this second syntax
% with proc being a struct array, the optional argument pvi (for
% parameter-value-index) must be specified. pvi specifies for each parameter
% which value is to be taken from the cell array .param.value.
% Also, there is the possibility to store output arguments that result
% from applying the transformation.
% 
% IN  
% fv   - struct of feature vectors
% proc - the actual transformation to apply. proc can be a string to be
%        evaluated by matlab's eval function, or a struct with fields
%        .eval  - a string that can be eval'ed (fv -> fv,
%                  or in the obsolete version: epo -> fv)
%        .param - a struct array with one entry for each free variable in
%                 proc.eval. with fields
%                 .var   - variable name (string)
%                 .value - cell array of values,
%                        when this is of length>1, the proc-transform
%                        has 'free parameters', otherwise all parameters
%                        are assigned
%        .memo - a cell array of variable names that should be stored
%                after having eval'ed the the transformation
%                string. If only one variable is to be saved, proc.memo
%                may just be a string. 
%                The stored variables will be saved in the proc.param
%                field, extending the list of free variables. Thus, the
%                returned and modified proc can be re-used later in
%                subsequent calls to proc_applyProc.
%                Variables contained in both proc.var and proc.memo will
%                *not* be stored.
%        .pvi - can be used to specify the parameter-value-index (see below)
%               instead of passing it as third input argument (used in
%               function XVALIDATION).
%
% pvi  - parameter-value-index: this argument is needed, when the
%        proc-transform has free parameters. pvi defines the
%        assignment from variables (.param.var) to values
%        (.param.value). Before eval'ing the transformation string,
%        variable proc.param(i).var will be assigned value
%        proc.param(i).value{pvi(i)}.
%
% OUT fv   - transformed struct of feature vectors
%     proc - as input, but the .param field is possibly extended by
%            the variables listed in proc.memo and the respective values.
%
% SEE select_proc, xvalidation

%% bb,as ida.first.fhg.de 07/2004

epo= fv;  %% for compatability
if ischar(proc),
  eval(proc);
  return;
end

if ~isfield(proc, 'param'),
  nParams = 0;
  nFreeParams= nParams;
else
  nParams= length(proc.param);
  if isfield(proc, 'memo'),
    nFreeParams= length(setdiff({proc.param.var}, proc.memo));
  else
    nFreeParams= nParams;
  end
end

if isfield(proc, 'pvi'),
  if nargin>=3,
    error('input argument pvi specified, although proc has a field .pvi');
  end
  pvi= proc.pvi;
elseif nargin<3,
  pvi= [];
end

if isempty(pvi),
  if prochasfreevar(proc),
    error('proc has free parameters, but input argument pvi undefined');
  end
elseif length(pvi)~=nFreeParams,
  error('Length of input argument pvi must match number of free variables');
end
%% Maybe proc_applyProc was called before and calculated the value of
%% some parameters (specified in field .memo) on training data (via the
%% code in field .train), and added them to the field .param.
%% Then (application on test data via the code in field .apply) we can
%% have more parameters (field .param) than we have indices for (pvi).
%% For those parameters (.memo) one specific value has been determined
%% so we have to one '1' as parameter-value-index pvi:
pvi(1,length(pvi)+1:nParams)= 1;

% The whole subsequent code is a bit paranoid: Make sure that the local
% variables used here have names that are most likely not contained in
% the proc string, so that we do not get any problems with overwriting.
% First, make copies of the input argruments
SAVEVARproc = proc;
SAVEVARnParams = nParams;
SAVEVARpvi = pvi;

if isfield(proc, 'sneakin') & ~isempty(proc.sneakin),
  for pp= 1:2:length(proc.sneakin),
    SAVEVARval= proc.sneakin{pp+1};
    SAVEVARcmd= sprintf('%s= SAVEVARval;', proc.sneakin{pp});
    eval(SAVEVARcmd);
  end
end

% Create all free variables and assign to their values index by pvi.
for SAVEVARpp= 1:SAVEVARnParams,
%   assignin('caller', SAVEVARproc.param(pp).var, SAVEVARproc.param(pp).value{pvi(pp)});
% $$$   val= proc.param(pp).value{pvi(pp)};
% $$$   cmd= sprintf('%s= val;', proc.param(pp).var);
  SAVEVARval= SAVEVARproc.param(SAVEVARpp).value{SAVEVARpvi(SAVEVARpp)};
  SAVEVARcmd= sprintf('%s= SAVEVARval;', proc.param(SAVEVARpp).var);
  eval(SAVEVARcmd);
end

eval(SAVEVARproc.eval);

% We have not yet saved the variables, so we still need to use the stupid
% long names
if nargout>1 & isfield(SAVEVARproc, 'memo'),
  % Handle that case that only one variable should be saved, and the
  % variable name is given as a string
  SAVEVARmem = SAVEVARproc.memo;
  if ~iscell(SAVEVARmem),
    SAVEVARmem= {SAVEVARmem};
  end
  % Variables contained both in free variable list and memo list will not
  % be saved
  if isfield(SAVEVARproc, 'param'),
    SAVEVARmem= setdiff(SAVEVARmem, {SAVEVARproc.param.var});
  end
  % Append the variables to be saved at the end of the param list
  SAVEVARpp = SAVEVARnParams+1;
  for SAVEVARmm = 1:length(SAVEVARmem),
    SAVEVARcmd= sprintf('%s', SAVEVARmem{SAVEVARmm});
    SAVEVARproc.param(SAVEVARpp).var= SAVEVARmem{SAVEVARmm};
    SAVEVARproc.param(SAVEVARpp).value= {eval(SAVEVARcmd)};
    SAVEVARpp= SAVEVARpp+1;
  end
end
