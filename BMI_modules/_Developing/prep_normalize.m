function [out] = prep_normalize(dat,varargin)
% prep_normalize (Pre-processing procedure):
%
% This function normalizes the feature vection. 
% (Input should be two-dimensional data)
%
% Example:
% [out] = prep_normalize(dat,{'Method','std'});
%
% Input:
%     dat - Feature vector, struct or data itself
% Options:
%     Method - 'std' (default)
%              'max'
%     Type - 'trial'   : Normalizing the data for each trial
%            'feature' : ... for each feature dimension (default)
% 
% Seon Min Kim, 05-2016
% seonmin5055@gmail.com

% Options of normalizing methods should be added later

opt = opt_cellToStruct(varargin{:});
if ~isfield(opt,'Method')
    opt.Method = 'std';
elseif ~isfield(opt,'Type')
    opt.Type = 'feature';
end

if isstruct(dat)
    if ~isfield(dat,'x')
        warning('OpenBMI: Data structure must have a field named ''x''');return
    end
    fv=dat.x;
elseif isnumeric(dat)
    fv=dat;
else
    warning('OpenBMI: Check for format of the data');return
end
z = size(fv);
if length(z)~=2
    warning('OpenBMI: Data should be in a form of "feature x trial"');return
end

switch opt.Method
    case 'std'
        if strcmp(opt.Type,'feature')
                c = repmat(std(fv,0,2),[1,z(2)]);
        elseif strcmp(opt.Type,'trial')
                c = repmat(std(fv,0,1),[z(1),1]);
        else
            warning('OpenBMI: Check for the ''Type'' option. It is either ''feature'' or ''trial''');return
        end
        fv = fv.*(1./c);
    case 'max'
        disp('OpenBMI: It might be updated soon')
    otherwise
        warning('OpenBMI: Unknown nomalizing method. It might be updated soon')
end

if isstruct(dat)
    out = rmfield(dat,'x');
    out.x = fv;
elseif isnumeric(dat)
    out = fv;
end
