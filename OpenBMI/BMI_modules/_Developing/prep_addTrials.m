function [out] = prep_addTrials(dat1, dat2)
% prep_addTrials (Pre-processing procedure):
%
% This function add the latter data(dat2) to the former data(dat1)
% 
% Example:
% [out] = prep_addTrials(dat1,dat2)
% [out] = prep_addTrials({dat1,dat2,dat3,dat4,...})
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

if iscell(dat1)
    temp = dat1{1};
    for i=2:size(dat1,2)
        temp = prep_addTrials(temp,dat1{i});
    end
    out = temp;
    return
end

if ~isfield(dat1,'x') || ~isfield(dat2,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if isfield(dat1,'chan') && isfield(dat2,'chan')
    if ~isequal(dat1.chan,dat2.chan)
        warning('OpenBMI: Unmatched channel')
        return
    end
end

dim1 = ndims(dat1.x);
dim2 = ndims(dat2.x);
if dim1~=dim2
    warning('OpenBMI: Unmatched data dimensions (Epoched or continuous)')
    return
end
switch dim1
    case 2
        if ~isequal(size(dat1.x,2),size(dat2.x,2))
            warning ('OpenBMI: Unmatched the number of channels')
            return
        end
    case 3
        if ~isequal(size(dat1.x,1),size(dat2.x,1))
            warning('OpenBMI: Unmatched data size')
            return
        elseif ~isequal(size(dat1.x,3),size(dat2.x,3))
            warning('OpenBMI: Unmatched the number of channels')
            return
        end
end


out.x = cat(dim1-1, dat1.x, dat2.x);

if isfield(dat1,'t') && isfield(dat2,'t')
    out.t = cat(2,dat1.t,dat2.t);
else
    warning('OpenBMI: Data should have a field named ''t''')
end

if isfield(dat1,'fs') && isfield(dat2,'fs')
    if dat1.fs == dat2.fs
        out.fs = dat1.fs;
    else
        warning('OpenBMI: Two input data have different frequency')
    end
end

if isfield(dat1,'y_dec') && isfield(dat2,'y_dec')
    out.y_dec = cat(2,dat1.y_dec,dat2.y_dec);
end

if isfield(dat1,'y_logic') && isfield(dat2,'y_logic')
    out.y_logic = cat(2,dat1.y_logic,dat2.y_logic);
end

if isfield(dat1,'y_class') && isfield(dat2,'y_class')
    out.y_class = cat(2,dat1.y_class,dat2.y_class);
end

if isfield(dat1,'class') && isfield(dat2,'class')
    if ~isequal(dat1.class,dat2.class)
        warning ('OpenBMI: Unmatched class')
        out.class = cat(1,dat1.class,dat2.class);   % Need to be modified. (for duplicated entries)
    else
        out.class = dat1.class;
    end
end
