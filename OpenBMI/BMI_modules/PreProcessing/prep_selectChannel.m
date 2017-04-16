function [ epo_ ] = prep_selectChannel( epo, idx )
%PROC_SELECT_CHANNEL Summary of this function goes here
%   Detailed explanation goes here
check_epo_param(epo);
epo_=epo;
epo_=rmfield(epo_,'x');
epo_=rmfield(epo_,'chan');

if isnumeric(idx)
  chans= idx;
else
  chans= sort(util_chanind(dat.clab,idx)); % sort to preserve original order in clab
end

if isfield(epo,'x')
    epo_.x=epo.x(:,:,chans);
else
    waring('epo.x is not exist')
end

if isfield(epo,'chan')
    epo_.chan=epo.chan(chans);
else
    waring('epo.chan is not exist')
end

end

