
function epo = proc_appendVoidClass(epo,voidClassName)
%
% Usage:
%         epo = proc_appendVoidClass(epo,voidClassName)
%
% Appends a void class to the input epo-struct.
% This function is for example used by qa_plotERP.m
%
% Simon Scholler, June 2011
%

if nargin<2
    voidClassName = '-';
end
void = epo;
void.x = NaN(size(epo.x,1),size(epo.x,2),1);
void.y = 1;
void.className = {voidClassName};
epo = proc_appendEpochs(epo,void);