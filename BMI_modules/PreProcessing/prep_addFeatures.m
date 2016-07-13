function [out] = prep_addFeatures(dat1,dat2)
% prep_addFeatures (Pre-processing procedure):
%
% This function add the latter feature vector(dat2) to the former one(dat1)
%
% Example:
% [out] = prep_addFeatures(dat1,dat2)
% [out] = prep_addFeatures({dat1,dat2,dat3,dat4,...})
% 
% Input:
%     dat1 - Feature vector in a struct form
%     dat2 - Feature vector in a struct form, with same class as dat1
%
% Returns:
%     out - Updated feature vector added aditional features
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if iscell(dat1)
    temp = dat1{1};
    for i=2:size(dat1,2)
        temp = prep_addFeatures(temp,dat1{i});
    end
    out = temp;
    return
end

if ~isfield(dat1,'x') || ~isfield(dat2,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if size(dat1.x,2) ~= size(dat2.x,2)
    warning('OpenBMI: Unmatched the number of trials')
    return
end
if ~isfield(dat1,'class') || ~isfield(dat1,'y_dec')
    warning('OpenBMI: Data structure should have a field named ''class'',''y_dec''')
end

r1 = rmfield(dat1,'x');
r2 = rmfield(dat2,'x');
if ~isequal(r1,r2)
    warning('OpenBMI: Two features must be extracted from the data with same class and trials')
    return
end

out = r1;
out.x = cat(1,dat1.x,dat2.x);
