function str= disp_proc(proc, varargin)
%str= disp_proc(proc, <opt>)
%
% IN   proc  - a fv-transform possibly having free variables,
%              see select_proc, proc_applyProc for details on the format
%      opt - propertylist and/or struct of options
%       .proc       - disp the eval string of proc
%       .param      - disp all params
%       .free_param - disp only free params (with more than 1 value)
%       .memo       - disp also the memo variables
%       .pvi        - parameter-value-index (vector [1 nParams]) to specify
%                     to parameter-value assignment, in case proc has
%                     free variables
%
% OUT   str  - if no output is given the string is printed to the terminal
% 
% SEE   select_proc, proc_applyProc, hasfreevar

%% bb ida.first.fhg.de 07/2004


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'proc', 1, ...
                  'param', 1, ...
                  'free_param', 0, ...
                  'memo', 0, ...
                  'pvi', [], ...
                  'lf', 1);

if isempty(opt.pvi) & isstruct(proc),
  opt.pvi= ones(1, length(proc.param));
end

if isstruct(proc),
  if opt.free_param,
    nVals= cell2mat(apply_cellwise({proc.param(:).value}, 'length'));
    pidx= find(nVals>1);
  else  
    pidx= 1:length(proc.param);
  end
else
  pidx= [];
end

if ~opt.memo & isstruct(proc) & isfield(proc, 'memo'),
  for pp= 1:length(pidx),
    if ismember(proc.param(pidx(pp)).var, proc.memo),
      pidx(pp)= 0;
    end
  end
  pidx= pidx(find(pidx));
end

str= '';
if opt.param & ~isempty(pidx),
  str= [str sprintf('%s=%s', proc.param(pidx(1)).var, ...
                    toString(proc.param(pidx(1)).value{opt.pvi(1)}))];
  for pp= 2:length(pidx),
    str= [str sprintf(', %s=%s', proc.param(pidx(pp)).var, ...
                      toString(proc.param(pidx(pp)).value{opt.pvi(pp)}))];
  end
  if opt.proc,
    str= [str sprintf(' -> ')];
  end
end

if opt.proc,
  if ischar(proc),
    str= [str sprintf('%s', proc)];
  elseif isfield(proc, 'eval'),
    str= [str sprintf('%s', proc.eval)];
  else 
    str= [str sprintf('Train: %s / Apply: %s', proc.train, proc.apply)];
  end
end

if opt.lf,
  str= [str sprintf('\n')];
end

if nargout==0,
  fprintf('%s', str);
end
