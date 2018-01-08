function [out] = prep_average(dat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_average (Pre-processing procedure):
%
% Synopsis:
%   [out] = prep_average(dat)
%
% Example :
%    [out] = prep_average(SMT)
%
% Arguments:
%     dat - Epoched data structure
%
% Returns:
%     out - Class-wise averaged data structure 
%
%
% Description:
%     Add specific channels to the former data(dat1) from dat2
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if ~isfield(dat,'y_dec') || ~isfield(dat,'y_logic') || ~isfield(dat,'y_class')
    warning('OpenBMI: Class information is missing')
    return
end

if ndims(dat.x)~=3 && size(dat.chan,2)~=1
    warning('OpenBMI: Data must be segmented')
end

n_cls = size(dat.y_logic,1);
tri_cls = cell(1,n_cls);
for cls = 1:n_cls
    tri_cls{cls} = mean(dat.x(:,(dat.y_dec==str2double(dat.class{cls,1})),:),2);
end
x = cat(2,tri_cls{1:end});

out = rmfield(dat,{'t','y_dec','y_logic','y_class'});
out.x = x;
