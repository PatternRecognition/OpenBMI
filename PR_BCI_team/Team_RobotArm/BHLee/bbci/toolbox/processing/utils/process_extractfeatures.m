function [fv_new,hlp] = process_extractfeatures(fv,dim,command,varargin)
%PROCESS_EXTRACTFEATURES is a utility function for combination. 
%
% usage:
%   [fv,hlp] = process_extractfeatures(fv,dim,command,<varnames,varvalues>);
%
% input:
%   fv      features
%   dim     the feature number used
%   command str to evaluate
%   varnames values of them should be given back 
%   varvalues  varnames should iniated with.
%
% output:
%   fv      feature
%   hlp     the hlp variable
%
% Guido Dornhege, 24/11/2004

if length(varargin)==2
  for i = 1:length(varargin{1})
    eval(sprintf('%s= varargin{2}{%i};',varargin{1}{i},i));
  end
end

fv_new = fv;

start = 0;
for i = 1:dim-1
  start = start+prod(fv.classifier_param{1}{i});
end

stop = start+prod(fv.classifier_param{1}{dim});
fv.x = fv.x(start+1:stop,:);

fv.x = reshape(fv.x,[fv.classifier_param{1}{dim},size(fv.x,2)]);

eval(command);

si = size(fv.x);
if size(fv_new.x,2)==1
  fv_new.classifier_param{1}{dim} = si(1:end);
  fv.x = fv.x(:);
else
  fv_new.classifier_param{1}{dim} = si(1:end-1);
  fv = proc_flaten(fv);
end


fv_new.x = cat(1,fv_new.x(1:start,:),fv.x,fv_new.x(stop+1:end,:));

if nargout>1
  if length(varargin)>=1
    hlp = cell(1,length(varargin{1}));
    for i = 1:length(hlp)
      eval(sprintf('hlp{%i} = %s;',i,varargin{1}{i})); 
    end
  else
    hlp = {};
  end
end

