function [out] = prep_addChannels(dat1, dat2, varargin)
% prep_addChannels (Pre-processing procedure):
%
% This function add data of specific channels from latter data(dat2) to the
% former data(dat1).
%
% Example:
% [out] = prep_addChannels(dat1,dat2,{'C3','C4'})
%
% Input:
%     dat1 - Data structure, continuous or epoched
%     dat2 - Data structure to be added to dat1
%     channels - Names of channels to be added in dat2, should be in a cell array
%
% Returns:
%     out - Updated data structure
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

out = rmfield(dat1,{'x','chan'});
ch = varargin{1};
out.chan = cat(2,dat1.chan,ch);     % It should be checked whether the channels are overlapped
ch_idx = find(ismember(dat2.chan,ch));
d1 = ndims(dat1.x);
if d1 ==1 || d1 == 2
    x = dat2.x(:,ch_idx);
elseif d1 == 3
    x = dat2.x(:,:,ch_idx);
else
    warning('Check for the dimension of input data')
end
out.x = cat(d1,dat1.x,x);