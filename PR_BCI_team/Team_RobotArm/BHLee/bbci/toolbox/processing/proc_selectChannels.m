function [dat2, W]= proc_selectChannels(dat, varargin)
%dat= proc_selectChannels(dat, chans)
%
% IN   dat   - data structure of continuous or epoched data
%      chans - cell array or list of channels to be selected,
%              see chanind for format
%
% OUT  dat   - updated data structure
%
% SEE chanind

% bb, ida.first.fhg.de

error(nargchk(2,inf,nargin));
if nargin==2 && length(varargin)==1 && ischar(varargin{1}) && ...
          (isempty(varargin{1}) || strcmp(varargin{1},'*')),
  dat2= dat;
  W= eye(size(dat.x,2));
  return;
end
if isnumeric(varargin{1})
  chans= varargin{1};
else
  chans= sort(chanind(dat.clab, varargin{:})); % sort to preserve original order in clab
end
if nargout>1,
  W= eye(length(dat.clab));
  W= W(:,chans);
end

if isfield(dat,'xUnit') && iscell(dat.xUnit)
  restdims = size(dat.x);
  dim1=1;
  for idx=1:length(restdims)
    dim1 = dim1 * restdims(idx);
    if dim1 == prod(dat.dim) 
      restdims = restdims((idx+1):end);
      break;
    end;
  end;
  dat.x = reshape(dat.x,[prod(dat.dim) restdims]);
end

dat2= rmfield(dat, 'x');
dat2.x= dat.x(:,chans,:);

if isfield(dat,'xUnit') && iscell(dat.xUnit)
  restdims = size(dat.x);
  restdims = restdims(2:end);
  dat.x = reshape(dat.x,[dat.dim restdims]);
  restdims2 = size(dat2.x);
  restdims2 = restdims2(2:end);
  dat2.x = reshape(dat2.x,[dat.dim restdims2]);
end

if isfield(dat,'clab')
  dat2.clab= dat.clab(chans);
end
if isfield(dat,'scale')
  dat2.scale= dat.scale(chans);
end
if isfield(dat,'p')
  dat2.p= dat.p(:,chans);
end
if isfield(dat,'V')
  dat2.V= dat.V(:,chans);
end
