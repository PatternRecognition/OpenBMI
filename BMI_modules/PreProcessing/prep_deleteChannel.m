function [ out ] = prep_deleteChannel(dat,varargin )
% prep_deleteChannel (Pre-processing procedure):
%
% Description:
%     This function deletes data of specific channels from the data.
%
% Example:
% out = prep_deleteChannel(dat,{'Name',{'C3','C4'}})
% out = prep_deleteChannel(dat,{'Index',[25,29]})
%
% Input:
%     dat - Continuous or segmented data structure
% Option:
%     'Name' or 'Index' - Name or index of channels to be deleted
% Return:
%     out - Data structure excluding specific channels
%
% Seon Min Kim, 07-2016
% seonmin5055@gmail.com

opt = opt_cellToStruct(varargin{:});

if isfield(opt,'Name') && isfield(opt,'Index')
    if find(ismember(dat.chan,opt.Name))~=opt.Index
        warning('OpenBMI: Mismatch between name and index of channels');return
    end
%     ch = opt.Name;
    idx = opt.Index;
elseif isfield(opt,'Name') && ~isfield(opt,'Index')
    ch = opt.Name;
    idx = find(ismember(dat.chan,ch));
    if size(idx)~=size(ch,2),warning('OpenBMI: Error in ''Name''');return;end
elseif ~isfield(opt,'Name') && isfield(opt,'Index')
    idx = opt.Index;
%     ch = dat.chan(idx);
else warning('OpenBMI: Channels should be specified in a correct form');return
end

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''');return
end
if ~isfield(dat,'chan')
    warning('OpenBMI: Data structure must have a field named ''chan''');return
end

out=dat;
out.chan(idx)=[];
if ndims(dat.x)==2
    out.x(:,idx)=[];
elseif ndims(dat.x)==3
    out.x(:,:,idx)=[];
else warning('OpenBMI: Check for the dimension of input data');return
end
