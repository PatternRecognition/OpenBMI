function [out] = prep_grandAverage(dat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_grandAverage
%
% Synopsis:
%   [out] = prep_grandAverage(dat,<var>)
%
% Example :
%   [out] = prep_grandAverage({dat1,dat2,...,datN});
%
% Arguments:
%     dat - Segmented data, structure or data itself, in a cell type
%
% Returns:
%     out - Data structure averaged across subjects
%
% Description:
%    This function averages the data across subjects. It is recommended that
%    input data be separated by each class in advance.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 01-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% chan, class °°ÀºÁö È®ÀÎ ÇÊ¿ä.
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
