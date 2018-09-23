function [out] = prep_selectClass(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_SELECTCLASS - selects data of specified classes from continuous or epoched data
% prep_selectClass (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_selectClass(DAT, <OPT>)
%
% Example :
%     out = prep_deleteClass(dat, {'right', 'left'});
%     out = prep_selectClass(dat, {'class', {'right', 'left'}});
%
% Arguments:
%     dat - Structure. Continuous data or epoched data (data.x)
%     varargin - struct or property/value list of optional properties:
%           class: Name of classes that you want to select (e.g. {'right', 'left'})
%           
% Returns:
%     out - Data structure which has selected class (continuous or epoched)
%
%
% Description:
%     This function selects data of specified classes 
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
elseif iscell(varargin{1})&& ~strcmpi(varargin{1}{1}, {'class'})
    opt.class = varargin{1};
else
    opt = opt_cellToStruct(varargin{:});
end

if ~all(isfield(dat, {'x', 'y_dec', 'y_class', 'y_logic', 'y_class', 'chan', 'class'}))
    error('OpenBMI: Data must have a field named ''x'', ''y_dec'', ''y_class'', ''y_logic'', ''y_class'', ''chan'', and ''class''');
end

cls_idx = ismember(dat.class(:,2), opt.class)';
cls_logical = any(dat.y_logic(cls_idx,:), 1);

if ndims(dat.x) == 3 || (ismatrix(dat.x) && length(dat.chan) == 1)
    out.x = dat.x(:, cls_logical, :);
end

out.t = dat.t(cls_logical);
out.y_dec = dat.y_dec(cls_logical);
out.y_logic = dat.y_logic(cls_idx, cls_logical);
out.y_class = dat.y_class(cls_logical);
out.class = dat.class(cls_idx, :);

out = opt_history(out, 'prep_selectClass', opt);

end