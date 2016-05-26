function [param,m,label] = adaptLDA(data,csp_w,init_param,m,varargin)

% This function classifies 'dat' using ini_param, parameters of LDA 
% and updates the parameters.
% 
% Example:
% [param,label] = adaptLDA(fv,csp_w,init_param,{'Method','pmean';'UC',0.03;'Nadapt',30});
%
% Input:
%     data  - Data (not a structure, data itself)
%     csp_w - Weight of CSP
%     init_param - Initial LDA parameters, with fields w and b
%     m     - Global mean of two classes
% Options:
%     Method - Adaptation method
%              'pmean' [Vidaurre et al, IEEE Trans Biomed Eng, 2011] (default)
%              http://dx.doi.org/10.1109/TBME.2010.2093133
%     UC - Update coefficient, learning rate
%          scalar value between [0,1] (default = 0.05)
%     Nadapt - Number of trials used for adaptation
% Returns:
%     param - Updated parameter structure
%     label - Output of adapted classifier
%
%
% Seon Min Kim, 05-2016
% seonmin5055@gmail.com

opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'Method')
    opt.Method='pmean';
end
if ~isfield(opt,'UC')
    opt.UC=0.05;
end
if ~isfield(opt,'Nadapt')
    opt.Nadapt=1;
end
if opt.Nadapt~=size(data,2)
    warning('OpenBMI:');return
end

%% Extracting features in test data
% [nt,ntr,nch]=size(data);
% fv=squeeze(log(var(reshape(reshape(data,[nt*ntr,nch])*csp_w,[nt,ntr,size(csp_w,2)]))));
fv = func_projection(data,csp_w);
fv = func_featureExtraction(fv,{'feature','logvar'});

%% Estimate the class label and update the parameters

switch opt.Method
    case 'pmean'
        m=init_param.cf_param.b\init_param.cf_param.w;
        label = func_predict(fv,init_param)';
%             switch size(data,2)
%                 case 1
%                     label = real(init_param.w'*fv' + init_param.b);
%                     % label(i) = func_predict(fv,init_param);
%                 otherwise
%                     label = real(init_param.w'*fv + init_param.b*ones(1,size(data,2)));
%             end
            label(label>=0)=2;label(label<0)=1;
        m = (1-opt.UC)*m+opt.UC*mean(fv,2);
        param=init_param;
        param.cf_param.b=-init_param.cf_param.w'*m;
    otherwise
        warning('OpenBMI: not implemented yet');return
end
