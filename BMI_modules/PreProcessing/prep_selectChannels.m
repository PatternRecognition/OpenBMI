function [out] = prep_selectChannels(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_selectChannels (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_selectChannels(DAT,<OPT>)
%
% Example :
%     out = prep_selectChannels(data, {'Name',{'Fp1', 'Fp2'}})
%     out = prep_selectChannels(data, {'Index',[1 2]})
%
% Arguments:
%     dat - Structure. Continuous data or epoched data
%         - Data which channel is to be selected        
%     varargin - struct or property/value list of optional properties:
%          : channels - 'cell Name' or 'index' of channels that you want to select
%           
% Returns:
%     out - Data structure which has selected channels (continuous or epoched)
%
%
% Description:
%     This function selects data of specified channels
%     from continuous or epoched data.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% minho_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    warning('OpenBMI: Channels should be specified')
    out = dat;
    return
end
opt = opt_cellToStruct(varargin{:});

if ~isfield(dat, 'chan')
    warning('OpenBMI: Data must have a field named ''chan''')
    return
end
if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end

if isfield(opt,'Name') && isfield(opt,'Index')
    if find(ismember(dat.chan,opt.Name))~=opt.Index
        warning('OpenBMI: Mismatch between name and index of channels')
        return
    end
    ch_idx = opt.Index;
elseif isfield(opt,'Name') && ~isfield(opt,'Index')
    ch_idx = find(ismember(dat.chan,opt.Name));
elseif ~isfield(opt,'Name') && isfield(opt,'Index')
    ch_idx = opt.Index;
else
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end

out = rmfield(dat,{'x','chan'});
out.chan = dat.chan(ch_idx);
d = ndims(dat.x);
if d==3
    out.x = dat.x(:,:,ch_idx);
elseif d==2
    out.x = dat.x(:,ch_idx);
else
    warning('OpenBMI: Check for the dimension of input data')
    return
end
