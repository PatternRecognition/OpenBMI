function [ epo_ ] = prep_deleteChannel( epo, idx )
%PROC_DELECT_CHANNEL Summary of this function goes here
%   Detailed explanation goes here
check_epo_param(epo)
epo_=epo;

if isnumeric(idx)
  chans=idx;
else
  chans= sort(util_chanind(dat.clab, idx)); % sort to preserve original order in clab
end

if isfield(epo_,'x')
    epo_.x(:,:,chans)=[];
else
    waring('epo.x is not exist')
end

if isfield(epo_,'chan')
    epo_.chan(chans)=[]; 
else
    waring('epo.chan is not exist')
end

end

