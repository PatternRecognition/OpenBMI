function [out] = prep_selectChannels(dat, varargin)
% prep_selectChannels (Pre-processing procedure):
%
% This function selects data of specified channels
% from continuous or epoched data.
%
% Example:
% dat.chan = {Fp1, Fp2, ... O1, O2};
% out = prep_selectChannels(data, {'Name',{'Fp1', 'Fp2'}})
% out = prep_selectChannels(data, {'Index',[1 2]})
%
% Input:
%     dat - Structure. Data which channel is to be selected
%     channels - Cell. Name or index of channels that you want to select
%
% Returns:
%     out - Updated data structure
%
%
% Seon Min Kim, 03-2016
% seonmin5055@gmail.com


if ~isfield(dat, 'chan')
    error('myApp:argChk','Input data should have a field named chSet')
end
if ~isfield(dat,'x')
    warning('Data is missing: Input data structure must have a field named ''x''')
    return
end

if iscell(varargin{1}{2})
    ch = varargin{1}{2};
    ch_idx = find(ismember(dat.chan,ch));
elseif isnumeric(varargin{1}{2})
    ch_idx = varargin{1}{2};
else
    error('myApp:argChk','Enter the channel information in a correct form')
end

out = rmfield(dat,{'x','chan'});
out.chan = dat.chan(ch_idx);
d = ndims(dat.x);
if d==3
    out.x = dat.x(:,:,ch_idx);
elseif d==2
    out.x = dat.x(:,ch_idx);
else
    warning('Check for the dimension of input data')
    return
end
