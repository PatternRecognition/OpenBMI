function [out] = prep_grandAverage(dat)
% prep_grandAverage (Pre-processing procedure):
%
% This function averages the data across subjects. It is recommended that
% input data be separated by each class in advance.
%
% Example:
% [out] = prep_grandAverage({dat1,dat2,...,datN});
%
% Input:
%     dat - Segmented data, structure or data itself, in a cell type
% Returns:
%     out - Data structure averaged across subjects
%
% Seon Min Kim, 05-2016
% seonmin5055@gmail.com

s = length(dat);
x = cell(1,s);
if isstruct(dat{1})
    if ~prod(cellfun(@(x) isfield(x,'x'),dat))
        warning('OpenBMI: Data structure must have a field named ''x''');return
    end
    EachDataSize=cellfun(@(y) size(y.x), dat,'UniformOutput',false);
    if ~isequal(EachDataSize{:})
        warning('OpenBMI: Data size must be same (Epoched or continuous)');return
    end
% chan, class 같은지 확인 필요.
    for i=1:s
        x{i} = dat{i}.x;
    end
    nd = ndims(dat{1}.x);
elseif isnumeric(dat{1}) && (ismatrix(dat{1}) || ndims(dat{1})==3)
    EachDataSize=cellfun(@size, dat,'UniformOutput',false);
    if ~isequal(EachDataSize{:})
        warning('OpenBMI: Data size must be same (Epoched or continuous)');return
    end
    x = dat;
    nd = ndims(dat{1});
else
    warning('OpenBMI: Check for format or dimension of the data');return
end

switch nd
    case 2
        Dat = cat(3,x{:});
        M = mean(Dat,3);
    case 3
        Dat = cat(4,x{:});
        M = mean(Dat,4);
end

if isstruct(dat{1})
    out = rmfield(dat{1},'x');
    out.x = M;
elseif isnumeric(dat{1})
    out = M;
end