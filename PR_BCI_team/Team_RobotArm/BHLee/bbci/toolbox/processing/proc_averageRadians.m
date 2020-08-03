function out= proc_averageRadians(epo, varargin)
%PROC_AVERAGERADIANS - Classwise calculated averages of radians values
%
%Synopsis:
% EPO= proc_averageRadians(EPO, <OPT>)
% EPO= proc_averageRadians(EPO, CLASSES)
%
%Input:
% EPO -      data structure of epoched data
%            (can handle more than 3-dimensional data, the average is
%            calculated across the last dimension)
% OPT struct or property/value list of optional arguments:
%  .std    - if true, standard deviation is calculated also 
%  .classes - classes of which the average is to be calculated,
%            names of classes (strings in a cell array), or 'ALL' (default)
%
%Output:
% EPO     - updated data structure with new field(s)
%  .N     - vector of epochs per class across which average was calculated
%  .std   - standard deviation, if requested (opt.std==1),
%           format as epo.x.

% Author(s): Benjamin Blankertz, 04-2005

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'classes', 'ALL', ...
                  'std', 0);

if ~isfield(epo, 'y'),
  warning('no classes label found: calculating average across all epochs');
  nEpochs= size(epo.x, ndims(epo.x));
  epo.y= ones(1, nEpochs);
  epo.className= {'all'};
end

if isequal(opt.classes, 'ALL'),
  classes= epo.className;
end
if ischar(classes), classes= {classes}; end
if ~iscell(classes),
  error('classes must be given cell array (or string)');
end
nClasses= length(classes);

if max(sum(epo.y,2))==1,
  warning('only one epoch per class - nothing to average');
  out= proc_selectClasses(epo, classes);
  out.N= ones(1, nClasses);
  return;
end

out= copy_struct(epo, 'not', 'x','y','className');
%  clInd= find(ismember(epo.className, classes));
%% the command above would not keep the order of the classes in cell 'ev'
evInd= cell(1,nClasses);
for ic= 1:nClasses,
  clInd= find(ismember(epo.className, classes{ic}));
  evInd{ic}= find(epo.y(clInd,:));
end

sz= size(epo.x);
TT= prod(sz(1:end-1));
out.x= zeros(TT, nClasses);
if opt.std,
  out.std= zeros(TT, nClasses);
end
out.y= eye(nClasses);
out.className= classes;
out.N= zeros(1, nClasses);
epo.x= reshape(epo.x, [TT sz(end)]);
for ic= 1:nClasses,
  if opt.std,
    [out.x(:,ic), outstd]= meanRadians(epo.x(:,evInd{ic}), 2);
    out.std(:,ic)= outstd;
  else
    out.x(:,ic)= meanRadians(epo.x(:,evInd{ic}), 2);
  end
  out.N(ic)= length(evInd{ic});
end

out.x= reshape(out.x, [sz(1:end-1) nClasses]);
if opt.std,
  out.std= reshape(out.std, [sz(1:end-1) nClasses]);
end
