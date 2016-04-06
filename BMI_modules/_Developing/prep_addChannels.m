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

if ~isfield(dat1,'x') || ~isfield(dat2,'x')
    warning('Data is missing: Input data structure must have a field named ''x''')
    return
end
if ~isfield(dat1,'chan') || ~isfield(dat2,'chan')
    warning('Channel information is missing: Input data structure must have a field named ''chan''')
    return
end
if isempty(varargin)
    warning('Data of all channels from the latter data will be added to the former data')
    ch = dat2.chan;
else
    ch = varargin{1};
end
s1=size(dat1.x);s2=size(dat2.x);
if s1(1:end-1)~=s2(1:end-1)
    warning('Unmatched data size')
    return
end

out = rmfield(dat1,{'x','chan'});

ch_ori = dat1.chan;
idx = [];
for i=1:size(ch_ori,2)
    for j=1:size(ch,2)
        if strcmp(ch_ori{i},ch{j});
            idx = [idx,j];
        end
    end
end
ch(idx) = [];

out.chan = cat(2,dat1.chan,ch);
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
