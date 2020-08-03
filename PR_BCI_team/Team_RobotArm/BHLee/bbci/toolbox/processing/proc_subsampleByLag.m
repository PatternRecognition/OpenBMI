function [dat, mrk]= proc_subsampleByLag(dat, lag, mrk)
%dat= proc_subsampleByLag(dat, lag)
%[dat, mrk]= proc_subsampleByLag(dat, lag, mrk)
%
% IN  dat  - data structure of continuous or epoched data
%     lag  - take each 'lag'th sample from input signals
%
% OUT dat  - updated data structure

% bb, ida.first.fhg.de


iv= ceil(lag/2):lag:size(dat.x,1);
dat.x= dat.x(iv,:,:);
dat.fs= dat.fs/lag;

if isfield(dat, 't'),
  dat.t= dat.t(iv);
end

if isfield(dat, 'T'),
  dat.T= dat.T./lag;
end

if nargin>2 && nargout>1,
  mrk.pos= ceil(mrk.pos/lag);
  mrk.fs= mrk.fs/lag;
end
