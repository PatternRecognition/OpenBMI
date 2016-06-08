function [ out ] = func_train( fv, varargin )
% func_train: Train a classifier
% 
% Example:
% [cls_param]=func_train(fv,{'classifier','LDA'});
% 
% Input:
%     fv - Feature vector to be trained
% Option:
%     classifier - 'LDA'
% Returns:
%     cls_param  - Structure of the classifier parameter
% 

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:}
end

switch lower(opt.classifier)
    case 'lda' %only binary class
%         out.cf_param=train_RLDAshrink(fv.x,fv.mrk.logical_y);
% BBCI toolbox
out.cf_param=train_RLDAshrink(fv.x,fv.y_logic);
        out.classifier='LDA';
end

end

