function [out,out_aa] = getClassifier(modus, varargin)
%function out = getClassifier(modus,...)
%   modus:
%     'init':
%          cls = getClassifier('init',cls)
%          cls as in bbci_bet_apply.
%     'apply':
% out =    getClassifier('apply',packetlength,cls)
%          packetlength in ms

% Guido, Matthias 2.12.2004
% TODO: extended documentation by Schwaighase
% ($Id: getClassifier.m,v 1.1 2006/04/27 14:22:08 neuro_cvs Exp $)

switch modus
 case 'init'
  %%%%%%%%%%%%%%%%%%
  % INIT %%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%
  cls = varargin{1};
  for c = 1:length(cls)
    % Since values could be changed, this is not needed any more
% $$$     % set defaults for scale, bias and integrate
% $$$     if ~isfield(cls(c),'scale')
% $$$       cls(c).scale = 1;
% $$$     end
% $$$     if ~isfield(cls(c),'bias')
% $$$       cls(c).bias = 0;
% $$$     end
% $$$     if ~isfield(cls(c),'integrate')
% $$$       cls(c).integrate = 1;
% $$$     end
    if ~isfield(cls(c),'condition')
      cls(c).condition = {};
    end
    if ~isfield(cls(c),'condition_param')
      cls(c).condition_param = {};
    end    
  end
  out = cls;
 case 'apply'
  %%%%%%%%%%%%%%%%%
  % APPLY %%%%%%%%%
  %%%%%%%%%%%%%%%%%
  cls = varargin{2};
  out = cell(1,length(cls));
  out_aa = cell(1,length(cls));
  packetlength = varargin{1};
  getFeature('reset');
  for c=1:length(cls)
    % check condition
    [flag, timeshift] = getCondition(cls(c).condition, ...
				     cls(c).condition_param,...
				     out(1:c-1), packetlength);
    
    if timeshift>0
      timeshift = 0;
    end
    
    if flag
      % evaluate the classifier
      
      % get the features
      fv = getFeature('apply',cls(c).fv,timeshift);
      % calculation of the values:
      out_aa{c} = feval(cls(c).applyFcn, cls(c).C, fv.x);
      out{c} = out_aa{c};
      %%% STANDARD POST_PROC %%%
      out = standardPostProc('apply_one',cls,out,c);
    
    else 
      % classifier does not need to be evaluated
      out{c} = NaN;
    end
  end
 otherwise 
  error('Unknown case')
end
  
return
