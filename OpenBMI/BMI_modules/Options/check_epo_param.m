function [ ] = check_epo_param( epo )
%CHECK_EPO_PARAM Summary of this function goes here
%   Detailed explanation goes here
if isfield(epo,'fs') && isfield(epo,'y') && isfield(epo,'y_logical') && isfield(epo,'class') ...
        isfield(epo,'t') && isfield(epo,'x') && isfield(epo,'chan')
    %     disp('All field of epo is exist');
end

if ~isfield(epo,'fs')
    warning('The field of epo.fs is not exist')
end
if ~isfield(epo,'y')
    warning('The field of epo.y is not exist')
end
if ~isfield(epo,'y_logical')
    warning('The field of epo.y_logical is not exist')
end
if ~isfield(epo,'class')
    warning('The field of epo.clss is not exist')
end
if ~isfield(epo,'t')
    warning('The field of epo.t is not exist')
end
if ~isfield(epo,'x')
    warning('The field of epo.x is not exist')
end
if ~isfield(epo,'chan')
    %     warning('The field of epo.chan is not exist')
end
end

