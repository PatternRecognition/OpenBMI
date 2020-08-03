function out = standardPostProc(modus, varargin)
% standardPostProc - The usual postprocessing for classifier outputs
%
% Synopsis:
%   standardPostProc(modus, arg1, arg2, ...)
%   standardPostProc('init', cls)
%   standardPostProc('init', cls, opt)
%   out = standardPostProc('apply', cls, out)
%   standardPostProc('cleanup')
%   
% Arguments:
%   modus: Post processing operation
%   cls: A classifier struct array in the format expected by
%       bbci_bet_apply. In particular, the fields .bias, .scale, and
%       .integrate are used by standardPostProc.
%   opt: Options structure or property/value list of options. Recognized
%       options are 'maxBufLength' (length of the classifier output
%       buffer for the integrate operation)
%   out: cell array of classifier outputs (cell array of column vectors)
%
% Returns:
%   out: processed classifier output (cell array of column vectors)
%   
% Description:
%   standardPostProc('init', cls, opt)
%       Initialize the routine (set up buffer for integrate operation)
% 
%   out = standardPostProc('apply', cls, out)
%       Apply postprocessing to the classifer output given by
%       'out'. Sequence of operations for classifier i is
%       - integration over the last cls(i).integrate classifier outputs
%       - add cls(i).bias
%       - multiply by cls(i).scale
%       - set values smaller than cls(i).dist to zero, correct for 
%         continuity, fix +1 and -1
%       - raise to the power of cls(i).alpha
%       - limit to cls(i).range
%       All these fields can be omitted, then the corresponding operation 
%       will not be performed.
%       Return the updated classifier output as a cell array of column
%       vectors.
%
%   standardPostProc('cleanup')
%       Clear integration buffer
%
%   
% See also: getClassifier,bbci_bet_apply
% 

% kraulem 20/01/05, Anton Schwaighofer
% ($Id: standardPostProc.m,v 1.2 2007/09/19 17:01:50 neuro_cvs Exp $)

persistent outBuf isInitialized

out = [];

modus = lower(modus);
if strcmp(modus, 'init'),
  if ~isempty(isInitialized) | isInitialized,
    warning('''init'' call without preceding ''cleanup''. Old data will be lost.');
  end
  cls = varargin{1};
  opt = propertylist2struct(varargin{2:end});
  opt = set_defaults(opt, 'maxBufLength', 100);
  % Test the bizarre case that somebody wants to integrate over more
  % than the buffer length:
  if isfield(cls,'integrate') & any([cls.integrate]>opt.maxBufLength),
    error('One or more classifiers have integration length larger than the buffer size');
  end
  if isfield(cls,'integrate') & (any([cls.integrate]<=0) | any(round([cls.integrate])~=[cls.integrate])),
    error('Invalid ''integrate'' parameter in classifiers');
  end
  % the outBuf contains unscaled/unbiased/unintegrated
  % cls_outputs. The most recent element is always at the end
  % (outBuf{c}(:,end) are the most recent classifier outputs)
  % initially, it is set to onedimensional NaNs.
  outBuf = cell(1,length(cls));
  for c = 1:length(cls)
    % initialize outBuf
    outBuf{c} = NaN*ones(1,opt.maxBufLength);
  end
  isInitialized = 1;
elseif isempty(isInitialized) | ~isInitialized,
  error('Call to standardPostProc.m without initialization');
else
  switch lower(modus)
    case 'apply'
      %%%%%%%%%%%%%%%%%
      % APPLY %%%%%%%%%
      %%%%%%%%%%%%%%%%%
      cls = varargin{1};
      out = varargin{2};
      for c=1:length(cls)
        % resize outBuf, if it has not yet been initialized otherwise (this
        % will happen in the first apply step)
        if length(out{c})>1
          out{c} = out{c}-mean(out{c});
        end
        if size(outBuf{c},1)<size(out{c},1)
          outBuf{c} = repmat(outBuf{c},[size(out{c},1), 1]);
        end
        % copy out to outBuf before doing any postprocessing
        outBuf{c} = [outBuf{c}(:,2:end) out{c}];
        % postprocessing light: 
        
        if isfield(cls(c),'bias') & ~isempty(cls(c).bias)
          out{c} = out{c}+cls(c).bias;
        end
        if isfield(cls(c),'scale') & ~isempty(cls(c).scale)
          out{c} = out{c}.*cls(c).scale;
        end
        
        if isfield(cls(c),'dist') & ~isempty(cls(c).dist)
          out{c} =  sign(out{c}).*max(0,(abs(out{c})-cls(c).dist)./(1-cls(c).dist));
        end
        
        if isfield(cls(c),'alpha') & ~isempty(cls(c).alpha)
          out{c} = sign(out{c}).*(abs(out{c}).^cls(c).alpha);
        end
        
        if isfield(cls(c),'integrate') & ~isempty(cls(c).integrate)
          out{c} = integrate_notisnan(outBuf{c}(:,(end-cls(c).integrate+1):end));
        end

        if isfield(cls(c),'range') & ~isempty(cls(c).range)
          out{c} = min(max(out{c},cls(c).range(1)),cls(c).range(2));
        end
        
      end
      
    case 'apply_one'
      %%%%%%%%%%%%%%%%%
      % APPLY %%%%%%%%%
      %%%%%%%%%%%%%%%%%
      cls = varargin{1};
      out = varargin{2};
      c = varargin{3};
      % resize outBuf, if it has not yet been initialized otherwise (this
      % will happen in the first apply step)
      if length(out{c})>1
        out{c} = out{c}-mean(out{c});
      end
      if size(outBuf{c},1)<size(out{c},1)
        outBuf{c} = repmat(outBuf{c},[size(out{c},1), 1]);
      end
      % copy out to outBuf before doing any postprocessing
      outBuf{c} = [outBuf{c}(:,2:end) out{c}];
      % postprocessing light: 
      if isfield(cls(c),'integrate') & ~isempty(cls(c).integrate)
        out{c} = integrate_notisnan(outBuf{c}(:,(end-cls(c).integrate+1):end));
      end
      
      if isfield(cls(c),'bias') & ~isempty(cls(c).bias)
        out{c} = out{c}+cls(c).bias;
      end
      if isfield(cls(c),'scale') & ~isempty(cls(c).scale)
        out{c} = out{c}.*cls(c).scale;
      end
      
      if isfield(cls(c),'dist') & ~isempty(cls(c).dist)
        out{c} =  sign(out{c}).*max(0,(abs(out{c})-cls(c).dist)./(1-cls(c).dist));
      end
      
      if isfield(cls(c),'alpha') & ~isempty(cls(c).alpha)
        out{c} = sign(out{c}).*(abs(out{c}).^cls(c).alpha);
      end
      
      if isfield(cls(c),'range') & ~isempty(cls(c).range)
        out{c} = min(max(out{c},cls(c).range(1)),cls(c).range(2));
      end
      
    case 'cleanup'
      outBuf = {};
      isInitialized = 0;
      
    otherwise
      error('Unknown operation')
  end
end

return

function out = integrate_notisnan(matrix)
% integrate over all values which are not NaN
out = nan*ones(size(matrix,1),1);
for i = 1:size(matrix,1)
  ind = find(~isnan(matrix(i,:)));
  if ~isempty(ind)
    out(i,1) = mean(matrix(i,ind));
  end
end
return
