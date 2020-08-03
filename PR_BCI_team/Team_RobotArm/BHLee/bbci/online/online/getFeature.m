function fv = getFeature(modus,varargin);
%GETFEATURE PRODUCES FEATURE VECTORS IN BBCI_BET_APPLY
%
% usage:
% ('init'):        getFeature('init',feature,opt,cont_proc);
% ('apply'):  fv = getFeature('apply',fn,timeshift);
% ('reset'):       getFeature('reset');
%
% Description:
%  'init'    initialize all important fields
%  'reset'   reset the workspace (clean calculated features)
%  'apply'   calculate feature if necessary and give back
%
% input:
%  feature     a feature struct array (see bbci_bet_apply)
%  fn          the number of features to use
%  timeshift   the timeshift to use (must be negative or zero)
%
% output:
%  fv          the concatenated feature
%
% see getClassifier and bbci_bet_apply for more informations
%
% Guido Dornhege, 02/12/04
% TODO: extended documentation by Schwaighase
% $Id: getFeature.m,v 1.2 2007/02/01 10:40:33 neuro_cvs Exp $

persistent feat feature commonFeat

switch modus
  case 'init'
    % INITIALIZATION
    feature = varargin{1};
    opt = varargin{2};
    cont_proc = varargin{3};
    
    feat = cell(1,length(feature));
    commonFeat = cell(1,length(feature));
    for i = 1:length(feature)
      if ~isfield(feature(i),'proc')
        feature(i).proc = [];
      end
      if ~isfield(feature(i),'proc_param');
        feature(i).proc_param = {};
      end
      if ~iscell(feature(i).proc)
        feature(i).proc = {feature(i).proc};
        feature(i).proc_param = {feature(i).proc_param};
      end
      commonFeat{i}.fs = opt.fs;
      %    commonFeat{i}.clab = cont_proc(i).clab;
      commonFeat{i}.clab = cont_proc(feature(i).cnt).clab;
    end
    
  case 'reset'
    % RESET
    feat = cell(1,length(feature));
    
  case 'apply'
    % APPLY
    fv = struct('x',[],'clab',{{}},'y',[]);
    fn = varargin{1};
    timeshift = varargin{2};
    if isempty(timeshift)
      timeshift = 0;
    end
    if timeshift > 0
      error('Timeshifts in the future do not make sense');
    end
    
    for i = 1:length(fn);
      f = commonFeat{fn(i)};
      % look if there is a feature calculated so far
      if ~isempty(feat{fn(i)})
        ind = find(ismember(feat{fn(i)}{1,:},timeshift));
        if ~isempty(ind)
          f = feat{fn(i)}{2,ind};
        end
      end
      if ~isfield(f,'x');
        f = commonFeat{fn(i)};
        % calculate feature
        f.x = storeContData('window',feature(fn(i)).cnt,feature(fn(i)).ilen_apply,timeshift);
        f.x(isnan(f.x))=0;
        f.t = (1-size(f.x,1))*1000/f.fs:1000/f.fs:0;
        % DO THE PROCESSING
        for j = 1:length(feature(fn(i)).proc)
          if ~isempty(feature(fn(i)).proc{j})
            f = feval(feature(fn(i)).proc{j},f,feature(fn(i)).proc_param{j}{:});
          end
        end
        
        % add to the list
        feat{fn(i)} = cat(2,feat{fn(i)},{timeshift;f});
      end
      
      % CONCATENATE
      f.x = f.x(:);
      if i==1
        fv = f;
      else
        fv.x = cat(1,fv.x,f.x);
      end
    end
  case 'update'
    fn = varargin{1};
    for ii = 2:2:length(varargin)
      feature(fn).(varargin{ii})= varargin{ii+1};
    end
  otherwise
    error('Unknown case');
end


return;
