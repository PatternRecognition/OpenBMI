function [ output ] = prep_selectTrials( dat, idx )
%PROC_SELECT_TRIALS Summary of this function goes here
%   Detailed explanation goes here
if isstruct(dat)
%     check_epo_param(dat);
    dat_=dat;
    dat_=rmfield(dat_,'x');
    dat_=rmfield(dat_,'y');
    dat_=rmfield(dat_,'y_logical');
    if ndims(dat.x)==3
        dat_.x=dat.x(:,idx,:);
        dat_.y=dat.y(:,idx);
        dat_.y_logical=dat.y_logical(:,idx);
    else
        waring('The dimension of epo.x should be three (DataxTrialsxChannels)')
    end
    output=dat_;
else
    dat=dat(:,idx,:); % Not structure,
    output=dat;
end

end

