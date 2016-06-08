function [ out ] = prep_selectClasses( dat, varargin )
% prep_selectClasses (Pre-processing procedure):
% 
% This function selects data of specified classes 
% from continuous or epoched data.
% 
% Example:
%     out = prep_selectClass(dat,{'Class',{'right', 'left','foot'}});
% 
% Input: 
%     dat - Structure. Data which classes are to be selected
%     classes - Cell. Name of classes that you want to select
% 
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if isempty(varargin)
    warning('OpenBMI: Classes should be specified');
    out = dat;
    return
end
opt=opt_cellToStruct(varargin{:});
if ~isfield(dat, 'class')
    warning('OpenBMI: Data structure must have a field named ''class''');
    return
end
if ~isfield(opt, 'Class')
    warning('OpenBMI: Classes should be specified in a correct form');
    return
end

cls_idx = find(ismember(dat.class(:,2),opt.Class));
tr_idx = find(ismember(dat.y_dec,cls_idx));
out = prep_selectTrials(dat,{'Index',tr_idx});
out.y_logic(~ismember(dat.class(:,2),opt.Class),:) = [];
