function [out] = prep_selectChannels(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_SELECTCHANNELS - select specific channels from continuous or epoched data
% prep_selectChannels (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_selectChannels(DAT,<OPT>)
%
% Example :
%     out = prep_selectChannels(data, {'Name',{'Fp1','Fp2'}})
%     out = prep_selectChannels(data, {'Index',[1 2]})
%     out = prep_selectChannels(data, {'Fp1','Fp2'})
%     out = prep_selectChannels(data, [1 2])
%     
% Arguments:
%     dat - Structure. Continuous data or epoched data (data.x)
%         - Data which channel is to be selected        
%     varargin - struct or property/value list of optional properties:
%           channels: 'index' or 'cell Name' of channels that you want to select
%           
% Returns:
%     out - Data structure which has selected channels (continuous or epoched)
%
% Description:
%     This function selects data of specified channels from continuous or epoched data.
%     Data can be not only whole of the struct, but also be a part of struct.
%     Continuous data should be [time * channels]
%     Epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = dat;

if isempty(varargin)
    warning('OpenBMI: Channels should be specified')
    return
end

if isnumeric(varargin{1})
    opt.index = varargin{1};
elseif iscell(varargin{1}) && ~any(strcmpi(varargin{1}{1}, {'name', 'index'}))
    opt.name = varargin{1};
else
    opt = opt_cellToStruct(varargin{:});    
end

if ~all(isfield(dat, {'chan', 'x'}))
    error('OpenBMI: Data must have a field named ''chan'' and ''x''')
end
if all(isfield(opt, {'index', 'name'}))
    warning('OpenBMI: selectChannels should be specified in a correct form')
    return
end 

if isfield(opt, 'index')
    opt.index = sort(opt.index);
    ch_idx = opt.index(opt.index <= length(dat.chan));
    if ~isequal(ch_idx, opt.index)
        warning('OpenBMI: Please check your channel configuration');
    end
elseif isfield(opt, 'name')
    ch_idx = find(ismember(dat.chan, opt.name));
    if ~all(ismember(opt.name, dat.chan(ch_idx)))
        warning('OpenBMI: Please check your channel configuration');
    end
else
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end
    
out.chan = dat.chan(ch_idx);

switch ndims(dat.x)
    case 2
        out.x = dat.x(:, ch_idx);
    case 3
        out.x = dat.x(:, :, ch_idx);
    otherwise
        warning('OpenBMI: Check for the dimension of input data');
    return
end

out = opt_history(out, 'prep_selectChannels', opt);

end