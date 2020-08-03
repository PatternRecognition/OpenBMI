function [epo, iArte]= proc_excludeArtifacts(epo, crit, clab)
%epo= proc_excludeArtifacts(epo, crit, <clab>)
%
% IN  epo  - structure of epoched data
%     crit - artifact rejection criteria, see find_artifacts
%            when this is a scalar value, it is used to set the
%            maxmin criterium
%     clab - labels of the channels to be checked for artifacts

if ~exist('clab','var'),
  clab= epo.clab;
end

if isreal(crit),
  crit= struct('maxmin',100);
end

iArte= find_artifacts(epo, clab, crit);
epo= proc_selectEpochs(epo, setdiff(1:size(epo.x,3),iArte));
