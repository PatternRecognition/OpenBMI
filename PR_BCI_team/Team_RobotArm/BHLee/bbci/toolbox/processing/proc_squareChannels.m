function dat= proc_squareChannels(dat, varargin)
%dat= proc_squareChannels(dat, chans)
%
% square all input signals samplewise.
%
% IN   dat   - data structure of continuous or epoched data
%      chans - cell array or list of channels, see chanind for possible
%              input formats, default all
%
% OUT  dat   - updated data structure
%
% SEE  chanind

% bb, ida.first.fhg.de


if nargin<2,
  chans= 1:size(dat.x,2);
else
  chans= chanind(dat, varargin{:});
end

dat.x(:,chans,:)= (dat.x(:,chans,:)).^2;

if isfield(dat, 'yUnit'),
  dat.yUnit= [dat.yUnit '^2'];
%  dat.yUnit= ['(' dat.yUnit ')^2'];
end
