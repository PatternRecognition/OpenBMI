function [out] = prep_deleteChannels(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_DELETECHANNELS - delete specific channels from continuous or epoched data
% prep_deleteChannels (Pre-processing procedure):
% 
% Synopsis:
%     [out] = prep_deleteChannels(DAT,<OPT>)
%
% Example:
%     out = prep_deleteChannels(data, {'name', {'Fp1','Fp2'}})
%     out = prep_deleteChannels(data, {'index', [1 2]})
%     out = prep_deleteChannels(data, {'Fp1', 'Fp2'})
%     out = prep_deleteChannels(data, [1 2])
%     
% Arguments:
%     dat - Structure. Continuous data or epoched data (data.x)
%         - Data which channel is to be deleted
%     varargin - struct or property/value list of optional properties:
%          : channels - 'index' or 'cell name' of channels that you want to delete
%           
% Returns:
%     out - Data structure excluding specific channels (continuous or epoched)
%
% Description:
%     This function delete specific channels from the data.
%     Data can be not only whole of the struct, but also be a part of struct.
%     Continuous data should be [time * channels]
%     Epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 09-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = dat;

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
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end

if isfield(opt, 'index')
    opt.index = sort(opt.index);
    ch_idx = opt.index(opt.index <= length(dat.chan));
    if ~isequal(ch_idx, opt.index)
        warning('OpenBMI: Please check your channel configuration');
    end
elseif isfield(opt, 'name')
    ch_idx = ismember(dat.chan, opt.name);
    if ~isequal(opt.name, dat.chan(ch_idx))
        warning('OpenBMI: Please check your channel configuration');
    end
else
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end

out.chan(ch_idx) = [];

switch ndims(dat.x)
    case 2
        out.x(:, ch_idx) = [];
    case 3
        out.x(:, :, ch_idx) = [];
    otherwise
        warning('OpenBMI: Check for the dimension of input data');
    return
end

out = opt_history(out, 'prep_deleteChannels', opt);

end