function [out] = prep_selectTrials(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_SELECTTRIALS - Selects data of specified trials from epoched data
% prep_selectTrials (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_selectTrials(DAT, <OPT>)
%
% Example :
%     out = prep_selectTrials(dat, [20:35]);
%     out = prep_selectTrials(dat, {'Index', [20:35]});
%
% Arguments:
%     dat - Structure. epoched data
%         - Data which trials are to be selected
%     varargin - struct or property/value list of optional properties:
%           index: index of trials to be selected
%
% Returns:
%     out - Data structure which has selected channels from epoched data
%
% Description:
%     This function selects data of specified trials from epoched data.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = dat;

if isempty(varargin)
    warning('OpenBMI: Trials should be specified');
    return
end

if isnumeric(varargin{1})
    opt.index = varargin{1};
elseif isstruct(varargin{1})
    opt = varargin{1};
else
    opt = opt_cellToStruct(varargin{:});
end

if ~all(isfield(dat, {'x', 't', 'y_dec', 'y_logic', 'chan', 'y_class'}))
    warning('OpenBMI: Data structure must have a field named ''x'', ''t'', ''y_dec'',''y_logic'', ''chan'', and ''y_class''');
    return
end

idx = ismember(1:size(dat.x, 2), opt.index);
switch ndims(dat.x)
    case 3
        x = dat.x(:, idx, :);
    case 2
        if size(dat.chan, 2) == 1
            x = dat.x(:, idx);
            warning('OpenBMI: just 1 channel data?');
        else
            x = permute(dat.x, [1 3 2]);
            warning('OpenBMI: just 1 trial data?');
        end
    case 1
        x = dat.x;
    otherwise
        warning('OpenBMI: Check for the data dimensionality')
        return;
end

out.x = x;
out.t = dat.t(idx);
out.y_dec = dat.y_dec(idx);
out.y_logic = dat.y_logic(:, idx);
out.y_class = dat.y_class(idx);

out = opt_history(out, 'prep_selectTrials', opt);

end