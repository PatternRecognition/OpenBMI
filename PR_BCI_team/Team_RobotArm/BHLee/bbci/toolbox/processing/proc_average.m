function out= proc_average(epo, varargin)
%PROC_AVERAGE - Classwise calculated averages
%
%Synopsis:
% EPO= proc_average(EPO, <OPT>)
% EPO= proc_average(EPO, CLASSES)
%
%Input:
% EPO -      data structure of epoched data
%            (can handle more than 3-dimensional data, the average is
%            calculated across the last dimension)
% OPT struct or property/value list of optional arguments:
%  .policy - 'mean' (default), 'nanmean', or 'median'
%  .std    - if true, standard deviation is calculated also 
%  .classes - classes of which the average is to be calculated,
%            names of classes (strings in a cell array), or 'ALL' (default)
%
% For compatibility PROC_AVERAGE can be called in the old format with CLASSES
% as second argument (which is now set via opt.classes):
% CLASSES - classes of which the average is to be calculated,
%           names of classes (strings in a cell array), or 'ALL' (default)
%
%Output:
% EPO     - updated data structure with new field(s)
%  .N     - vector of epochs per class across which average was calculated
%  .std   - standard deviation, if requested (opt.std==1),
%           format as epo.x.

% Author(s): Benjamin Blankertz, long time ago

%% delegate a special case:
if isfield(epo, 'yUnit') && isequal(epo.yUnit, 'dB'),
% warning('Using ''proc_dBaverage''. In future this warning will be skipped.');
  out= proc_dBaverage(epo, varargin{:});
  return;
end

if nargin==2
  opt.classes = varargin{:};
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'policy', 'mean', ...
                  'classes', 'ALL', ...
                  'std', 0);
		  
classes = opt.classes;

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
out.x= zeros(prod(sz(1:end-1)), nClasses);
if opt.std,
  out.std= zeros(prod(sz(1:end-1)), nClasses);
  if exist('mrk_addIndexedField')==2,
    %% The following line is only to be executed if the BBCI Toolbox
    %% is loaded.
    out= mrk_addIndexedField(out, 'std');
  end
end
out.y= eye(nClasses);
out.className= classes;
out.N= zeros(1, nClasses);
epo.x= reshape(epo.x, [prod(sz(1:end-1)) sz(end)]);
for ic= 1:nClasses,
  switch(lower(opt.policy)),  %% alt: feval(opt.policy, ...)
   case 'mean',
    out.x(:,ic)= mean(epo.x(:,evInd{ic}), 2);
   case 'nanmean',
    out.x(:,ic)= nanmean(epo.x(:,evInd{ic}), 2);
   case 'median',
    out.x(:,ic)= median(epo.x(:,evInd{ic}), 2);
   otherwise,
    error('unknown policy');
  end
  if opt.std,
    if strcmpi(opt.policy,'nanmean'),
      out.std(:,ic)= nanstd(epo.x(:,evInd{ic}), 0, 2);
    else
      out.std(:,ic)= std(epo.x(:,evInd{ic}), 0, 2);
    end
  end
  out.N(ic)= length(evInd{ic});
end

out.x= reshape(out.x, [sz(1:end-1) nClasses]);
if opt.std,
  out.std= reshape(out.std, [sz(1:end-1) nClasses]);
end
