function epoj= proc_joinClasses(epo, varargin)
%PROC_JOINCLASSES - joins together different classes
%
%Synopsis:
% EPO= proc_joinClasses(EPO, CLASSES)
% EPO= proc_joinClasses(EPO, Prop1, Value1, ...)
%
%Arguments:
% EPO - feature vector date structure
% CLASSES - cell array, each cell specifying a group of classes to be joint.
%       Each group can be defined as vector of class indices or as cell array
%       of class names.
%
%Properties:
% 'classes': 
%
%Output:
% EPO - structure with joint classes

if length(varargin)==1,
  opt= struct('classes', varargin{1});
  isdefault.className= 1;
else
  opt= propertylist2struct(varargin{:});
  [opt, isdefault]= ...
      set_defaults(opt, ...
                   'classes', {1:size(epo.y,1)}, ...
                   'className', '');
end

if ~iscell(opt.classes),
  opt.classes= {opt.classes};
end

if max(sum(epo.y,2))>1,
  epo= proc_average(epo);
end

leaveOut= {'x','y','className'};
if isfield(opt, 'indexedByEpochs'),
  leaveOut= cat(2, leaveOut, {'indexedByEpochs'}, epo.indexedByEpochs);
end
epoj= copy_struct(epo, 'not', leaveOut{:});

nNewClasses= length(opt.classes);
szx= size(epo.x);
epoj.x= zeros([szx(1) szx(2) nNewClasses]);
epoj.y= eye(nNewClasses);
if isdefault.className,
  epoj.className= cell(1, nNewClasses);
else
  epoj.className= opt.className;
end
for k= 1:nNewClasses,
  clInd= getClassIndices(epo, opt.classes{k});
  idx= find(any(epo.y(clInd,:),1));
  epoj.x(:,:,k)= mean(epo.x(:,:,idx), 3);
  if isdefault.className,
    epoj.className{k}= vec2str(epo.className(opt.classes{k}), '%s', '+');
  end
end
