function [out] = prep_addChannels(dat1, dat2, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_ADDCHANNELS - add specific channels to the former data(dat1) from dat2
% prep_addChannels (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_addChannels(DAT, DAT2, <OPT>)
%
% Example:
%     out = prep_addChannels(dat1, dat2, {'name', {'Fp1','Fp2'}})
%     out = prep_addChannels(dat1, dat2, {'index', [1 2]})
%     out = prep_addChannels(dat1, dat2, {'Fp1', 'Fp2'})
%     out = prep_addChannels(dat1, dat2, [1 2])
%
% Arguments:
%     dat1 - Structure. Continuous data or epoched data (data.x)
%     dat2 - Structure which should have specific channels. Continuous data or epoched data (data.x)
%     varargin - struct or property/value list of optional properties:
%           'Name'- Cell type. channel name in dat2 to be added to dat1
%           'Index'- channel index in dat2 to be added to dat1
%
% Returns:
%     out - Data structure which channels are added (continuous or epoched)
%
% Description:
%     Add specific channels to the former data(dat1) from dat2.
%     Continuous data should be [time * channels]
%     Epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 09-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out = dat1;
if isempty(varargin)
    warning('OpenBMI: Data of all channels from the latter data will be added to the former data')
    opt.name = dat2.chan;
else
    if isnumeric(varargin{1})
        opt.index = varargin{1};
    elseif iscell(varargin{1})&& ~any(strcmpi(varargin{1}{1}, {'name', 'index'}))
        opt.name = varargin{1};
    else
        opt = opt_cellToStruct(varargin{:});    
    end
end

if ~all(cellfun(@(x) all(isfield(x, {'x', 'chan'})), {dat1, dat2}))
    error('OpenBMI: Data must have a field named ''chan'' and ''x''')
end

if all(isfield(opt, {'index', 'name'}))
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end

s1 = size(dat1.x);
s2 = size(dat2.x);

if s1(1:end - 1) ~= s2(1:end - 1)
    warning('OpenBMI: Unmatched data size')
    return
end

if isfield(opt, 'index')
    opt.index = sort(opt.index);
    ch_idx = opt.index(opt.index <= length(dat2.chan));
    if ~isequal(ch_idx, opt.index)
        warning('OpenBMI: Please check your channel configuration');
    end
elseif isfield(opt, 'name')
    ch_idx = ismember(dat2.chan, opt.name);
    if ~isequal(opt.name, dat2.chan(ch_idx))
        warning('OpenBMI: Please check your channel configuration');
    end
else
    warning('OpenBMI: Channels should be specified in a correct form')
    return
end
    
idx = ~ismember(dat2.chan(ch_idx), dat1.chan);
out.chan = horzcat(dat1.chan, dat2.chan(idx));
d1 = ndims(dat1.x);

switch d1
    case 2
        out.x = horzcat(dat1.x, dat2.x(:, ch_idx(idx)));
    case 3
        out.x = cat(d1, dat1.x, dat2.x(:, :, ch_idx(idx)));
    otherwise
        warning('OpenBMI: Check for the dimension of input data');
    return
end

out = opt_history(out, 'prep_addChannels', opt);

end