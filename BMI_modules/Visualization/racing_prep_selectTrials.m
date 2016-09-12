function [out] = racing_prep_selectTrials(dat,varargin)
% prep_selectTrials (Pre-processing procedure):
% 
% This function selects data of specified trials 
% from continuous or epoched data.
% 
% Example:
%     out = prep_selectTrials(dat,{'Index',[20:35]});
%     out = prep_selectTrials(dat,{'Time',[20:35]});
% Option for time interval is not considered yet, it should be added soon
% 
% Input: 
%     dat - Structure. Data which trials are to be selected
%     index - index of trials to be selected
%     time - time interval of trials to be selected
% 
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if isempty(varargin)
    warning('OpenBMI: Trials should be specified');
    out = dat;
    return
end
opt=opt_cellToStruct(varargin{:});
if ~isfield(dat, 'x')
    warning('OpenBMI: Data structure must have a field named ''x''');
    return
end
if ~isfield(dat, 't')
    warning('OpenBMI: Data structure must have a field named ''t''');
    return
end
if ~isfield(dat, 'y_dec') || ~isfield(dat, 'y_logic') || ~isfield(dat, 'y_class')
    warning('OpenBMI: Data structure must have a field named ''y_dec'',''y_logic'',''y_class''');
    return
end

idx = opt.Index;
nd = ndims(dat.x);
if nd == 3
    x = dat.x(:,idx,:);
elseif nd ==2 || nd ==1
    x = dat.x;
else
    warning('OpenBMI: Check for the data dimensionality')
    return
end
out = rmfield(dat,{'x','t','y_dec','y_logic','y_class'});
out.x = x;
out.t = dat.t(idx);
out.y_dec = dat.y_dec(idx);
out.y_logic = dat.y_logic(:,idx);
out.y_class = dat.y_class(idx);
