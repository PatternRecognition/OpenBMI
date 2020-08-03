function dat= proc_rectifyChannels(dat, varargin)
%dat= proc_rectifyChannels(dat, chans)
%
% convert all input signals samplewise to absolute values.
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

dat.x(:,chans,:)= abs(dat.x(:,chans,:));
