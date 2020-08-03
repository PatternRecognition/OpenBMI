function [cnt,apply] = proc_spatial_filtering_augcog(cnt,spatial);
%
% usage:
%   cnt = proc_spatial_filering_augcog(cnt,spatial);
%
% input:
%    cnt    a usual cnt/epo structure
%    spatial  -  a spatial filter:
%                 'none' : no spatial filter
%             LAPLACIAN:
%                 'diagonal': diagonal laplace filter
%                 'small': small laplace filter
%                 'large': large laplace filter
%                 'horizontal': horizontal laplace filter
%                 'vertical': vertical laplace filter
%                 'eight': use all eight channels around for laplacing
%               all of this can also be used as {'laplace',...} or 
%               {'laplace',...,remainChans}, where remainChans says 
%               which channels should remain (can also be 'filter all'). 
%               For ... you can also use 2-cell arrays (grid/laplace)
%               see proc_laplace for more documenation
%
%             COMMON AVERAGE REFERENCE:
%                  'car': common average reference
%                 or {'car',refChans} where refChans are the refChans 
%                                     (default all)
% 
%             COMMON MEDIAN REFERENCE
%                  'cmr': common median reference 
%                 or {'cmr','refChans}  where refChans are the referene channels (default all)
%
%             COMMON SPATIAL PATTERNS
%                  'csp': common spatial patterns
%                 or {'csp',nPat}  nPat is the number of used patterns per class%                 or {'csp',nPat,flag} where flag is true, is CSP should be really done (overfitting!!!) or should be prepared.
%
% output:
%    cnt    a usual cnt/epo structure, or -1 if a cnt structure was given, but epo structure (CSP) was required.
%
% Guido DOrnhege, 27/04/2004


if ~exist('spatial','var') | isempty(spatial)
  spatial = 'none';
end


if iscell(spatial)
  switch spatial{1}
   case 'none' 
    % nothing to do
   case 'laplace'
    if length(spatial)<3
      cnt = proc_laplace(cnt,spatial{2:end});
      apply.fcn = 'proc_laplace';
      apply.param = spatial(2:end);
    else
      cnt = proc_laplace(cnt,spatial{2},[],spatial{3});
      apply.fcn = 'proc_laplace';
      apply.param = {spatial{2},[],spatial{3:end}};
    end
   case {'diagonal','small','large','horizontal','vertical','eight'}
    cnt = proc_laplace(cnt,spatial{1},[],spatial{2:end});
    apply.fcn = 'proc_laplace';
    apply.param = {spatial{1},[],spatial{2:end}};
   case 'car'
    cnt = proc_commonAverageReference(cnt,spatial{2:end});
    apply.fcn = 'proc_commonAverageReference';
    apply.param = spatial(2:end);
   case 'cmr'
    cnt = proc_commonMedianReference(cnt,spatial{2:end});
    apply.fcn = 'proc_commonMedianReference';
    apply.param = spatial(2:end);
% $$$    case 'csp'
% $$$     if length(spatial)==3 & spatial{3}==false
% $$$       cnt.proc = sprintf('fv=proc_csp(epo,%i);fv=proc_logarithm(proc_variance(fv));',spatial{2});
% $$$     else
% $$$       if ndims==2
% $$$         cnt = -1;
% $$$       else
% $$$         cnt = proc_csp(cnt,spatial{2:min(2,length(spatial))});
% $$$       end
% $$$       warning('csp is label dependent, and can result in overfitting');
% $$$     end
  end
else
  switch spatial
   case 'none' 
    % nothing to do
   case {'diagonal','small','large','horizontal','vertical','eight'}
    cnt = proc_laplace(cnt,spatial);
    apply.fcn = 'proc_laplace';
    apply.param = spatial;
   case 'car'
    cnt = proc_commonAverageReference(cnt);
    apply.fcn = 'proc_commonAverageReference';
    apply.param = [];
   case 'cmr'
    cnt = proc_commonMedianReference(cnt);
    apply.fcn = 'proc_commonMedianReference';
    apply.param = [];
% $$$    case 'csp'
% $$$     if ndims==2
% $$$       cnt = -1; 
% $$$     else
% $$$       cnt = proc_csp(cnt,1); 
% $$$       warning('csp is label dependent, and can result in overfitting');
% $$$     end
  end
end
