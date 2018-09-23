function [out] = prep_deleteClass(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_REMOVECLASS - Delete data of specified classes from continuous or epoched data
% prep_removeClass (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_deleteClass(DAT, <OPT>)
%
% Example :
%     out = prep_deleteClass(dat, {'right', 'left'});
%     out = prep_deleteClass(dat, {'class', {'right', 'left'}});
%
% Arguments:
%     dat - Structure. Continuous data or epoched data (data.x)
%     varargin - struct or property/value list of optional properties:
%           class: Name of classes that you want to delete (e.g. {'right', 'left'})
%
% Returns:
%     out - Data structure which has deleted class (continuous or epoched)
%
% Description:
%     This function delete data of specific classes
%     from continuous or epoched data.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 09-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = dat;

if isempty(varargin)
    warning('OpenBMI: Class should be specified')
    return
end

if ischar(varargin{1})
    opt.class = {varargin{1}};
elseif iscell(varargin{1}) && ~strcmpi(varargin{1}{1}, {'class'})
    opt.class = varargin{1};
else
    opt = opt_cellToStruct(varargin{:});
end

if ~all(isfield(dat, {'x', 'y_dec', 'y_class', 'y_logic', 'y_class', 'class'}))
    error('OpenBMI: Data must have a field named ''x'', ''y_dec'', ''y_class'', ''y_logic'', ''y_class'', and ''class''');
end

cls_idx = ~ismember(dat.class(:, 2), opt.class)';
cls_logical = any(dat.y_logic(cls_idx, :), 1);

if ndims(dat.x) == 3 || (ismatrix(dat.x) && length(dat.chan) == 1)
    out.x = dat.x(:, cls_logical, :);
end

if isfield(dat, 't')
    out.t = dat.t(cls_logical);
end
if isfield(dat, 'y_dec')
    out.y_dec = dat.y_dec(cls_logical);
end
if isfield(dat, 'y_logic')
    out.y_logic = dat.y_logic(cls_idx,cls_logical);
end
if isfield(dat, 'y_class')
    out.y_class = dat.y_class(cls_logical);
end
if isfield(dat, 'class')
    out.class = dat.class(cls_idx, :);
end

out = opt_history(out, 'prep_deleteClass', opt);

end