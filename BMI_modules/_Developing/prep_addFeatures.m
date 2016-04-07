function [out] = prep_addFeatures(dat1,dat2)
% prep_addFeatures (Pre-processing procedure):
%
% This function add the latter feature vector(dat2) to the former one(dat1)
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

if ~isfield(dat1,'x') || ~isfield(dat2,'x')
    warning('Data is missing: Input data structure must have a field named ''x''')
    return
end

r1 = rmfield(dat1,'x');
r2 = rmfield(dat2,'x');
if ~isequal(r1,r2)
    warning('Two features must be extracted from the data with same class and trials')
    return
end

out = r1;
out.x = cat(1,dat1.x,dat2.x);
