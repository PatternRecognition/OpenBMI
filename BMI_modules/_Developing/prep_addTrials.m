function [out] = prep_addTrials(dat1, dat2)
% prep_addTrials (Pre-processing procedure):
%
% This function add the latter data(dat2) to the former data(dat1)
%
% Example:
% [out] = prep_addTrials(dat1,dat2)
%
% Input:
%     dat1 - Data structure, continuous or epoched
%     dat2 - Data structure to be added to dat1
%
% Returns:
%     out - Updated data structure
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if ~isfield(dat1,'x') || ~isfield(dat2,'x')
    warning('Data is missing: Input data structure must have a field named ''x''')
    return
end

dim1 = ndims(dat1.x);
dim2 = ndims(dat2.x);

if dim1~=dim2
    warning('Data dimensions are not same: Epoched or continuous')
    return
end

switch dim1
    case 2
        if ~isequal(size(dat1.x,2),size(dat2.x,2))
            warning ('Unmatched the number of channels')
            return
        end
    case 3
        if ~isequal(size(dat1.x,1),size(dat2.x,1))
            warning('Unmatched data size')
            return
        elseif ~isequal(size(dat1.x,3),size(dat2.x,3))
            warning('Unmatched the number of channels')
            return
        end
end

if isfield(dat1,'chan') && isfield(dat2,'chan')
    if ~isequal(dat1.chan,dat2.chan)
        warning('Unmatched channel')
        return
    else
        out.chan = dat1.chan;
    end
else
    warning('Channel information is missing: Input data should have a field named ''chan''')
end

out.x = cat(dim1-1, dat1.x, dat2.x);

if isfield(dat1,'t') && isfield(dat2,'t')
    out.t = cat(2,dat1.t,dat2.t);
else
    warning('Time information is missing: Input data should have a field named ''t''')
end

if isfield(dat1,'fs') && isfield(dat2,'fs')
    if dat1.fs == dat2.fs
        out.fs = dat1.fs;
    else
        warning('Two input data have different frequency')
    end
end

if isfield(dat1,'y_dec') && isfield(dat2,'y_dec')
    out.y_dec = cat(2,dat1.y_dec,dat2.y_dec);
end

if isfield(dat1,'y_logic') && isfield(dat2,'y_logic')
    out.y_logic = cat(2,dat1.y_logic,dat2.y_logic);
end

if isfield(dat1,'class') && isfield(dat2,'class')
    if ~isequal(dat1.class,dat2.class)
        warning ('Unmatched class')
        out.class = cat(1,dat1.class,dat2.class);   % Need to be modified. (for duplicated entries)
    else
        out.class = dat1.class;
    end
end

end