function [p,t,stats,terms,arg] = rmanova(dat,varargin)
% RMANOVA - performs a conventional n-way analysis of variance (ANOVA)
% or a repeated-measures ANOVA. A conventional ANOVA assumes a
% between-subjects design (different groups), a RM ANOVA a within-subjects
% design (all subjects participated in all conditions).
%
%Usage:
% [p,t,stats,terms,arg] = rmanova(dat,<OPT>)
% [p,t,stats,terms,arg] = rmanova(dat,varnames,<OPT>)
%
%Arguments:
% DAT      -  N*F1*F2*F3*...*FN data matrix. 
%             First dimension (rows): refers to the subjects, ie each row 
%             contains the data of one subject. The second and successive 
%             dimensions contain the
%             factors, with the size of the dimension being the number of
%             levels of that factor. Example: for 13 subjects and the 
%             factors Speller with 3 levels (Hex,Cake,Center speller) and
%             Attention with 2 levels (overt, covert), the data matrix
%             would have a size of 13*3*2. The entry dat(5,2,1) refers to
%             the fifth subject for Cake Speller (2nd level of Speller) and
%             overt attention (1st level of Attention).
%
% OPT - struct or property/value list of optional properties:
% 'varnames'   - CELL ARRAY of one or more factors (eg {'Speller'
%              'Attention'}). The order of the factors must correspond
%              to the order of the factors in the DAT matrix.
% 'design' -  Test design. 'independent' performs a conventional ANOVA. 
%             If 'repeated-measures' (default), performs a
%             repeated-measures ANOVA. In the latter case, Subject is
%             included as a random (ie not fixed) effect.
% 'display'  - if 'on' displays a table with the results (default 'on')
%
% Assumptions:
%   ANOVA   - homogeneity of variance: variances within each group are
%   equal
%   RM ANOVA - sphericity. Variation of population >difference< scores are
%   the same for all differences (for >2 levels of a factor).
%
% All other options are passed to the 'anovan' function.
%
%Returns:
% [p,t,stats,terms]     - 'help anovan' for details
% arg                   - arguments passed to anovan
%
% See also ANOVAN, PLOT_STATS.
%
% Note: If your experiment involves repeated-measures (your subject was run in all
% subconditions) you should use repeated-measures ANOVA (RM-ANOVA), because the
% assumption of independence of samples is violated. Furthermore, RM-ANOVA
% accounts for inter-subject variability and thus has more statistical
% power.

% Author(s): Matthias Treder 2011

varnames = [];
if nargin==2 
  if iscell(varargin{1})
    varnames = varargin{1};
  else varnames = {varargin{1}};
  end
  varargin={};
elseif  nargin>1 && iscell(varargin{1})
  varnames = varargin{1};
  varargin = varargin(2:end);
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'alpha',.05, ...
                 'display','on', ...
                 'model', 'full', ...
                 'sstype', 3, ...
                 'varnames', varnames, ...
                 'table',1,...
                 'verbose',1,...
                 'alpha',.05,...
                 'design', 'repeated-measures');

ss = size(dat);
nsbj = ss(1);           % Number of subjects
level = ss(2:end);

%% Factors and levels
% Number of factors
if ~isempty(opt.varnames)
  nfac = numel(opt.varnames);
  if nfac ~= ndims(dat)-1
    error('Number of factors %d does not match factors in data %d',nfac,ndims(dat)-1)
  end
else
  nfac = ndims(dat)-1;
end
               
% If no factor names provided, use default names
if isempty(opt.varnames)
  vn = num2cell(1:nfac);vn = apply_cellwise(vn,inline('num2str(x)'));
  opt.varnames = strcat('X',vn);
end


%% Check (RM) ANOVA assumptions // TODO


if strcmp(opt.design,'independent')
  % Homogenity of variance
  % TODO
elseif strcmp(opt.design,'repeated-measures')
  % Sphericity
end

%% ANOVA or rmANOVA
if opt.verbose
  names = cell2mat(strcat(opt.varnames,',')); names=names(1:end-1);
  fprintf('Performing a %d-way %s ANOVA with factors {%s} and %s = %d levels.\n',...
    nfac,opt.design,names,vec2str(level,'%d','x'),prod(level))
end


if strcmp(opt.design,'independent')
  % ANOVA model
  design = orthogonal_design(nsbj,opt.level);

elseif strcmp(opt.design,'repeated-measures')
  % Repeated measures ANOVA
  opt.varnames = {'Subject' opt.varnames{:}};  % Add subject as a factor
  if isfield(opt,'random')   % Add subject as random effect
    opt.random = [1 opt.random+1];
  else
    opt.random = 1;
  end
  design = orthogonal_design(1,[nsbj level]);
  % Specify to-be-tested interactions by hand to omit Subject  
  model = double(flipud(dec2bin(0:2^(nfac+1)-1))-'0');
  model( model(:,1) & sum(model(:,2:end),2)>0 , :) = []; % Omit all interactions involving Subject
  maineffects = find(sum(model,2)==1);
  interactions = find(sum(model,2)>1);
  opt.model = model([maineffects; interactions],:); % Bring in right order
else
  error('Unknown design ''%s''',opt.design)
end

%% Perform ANOVA
arg = struct2propertylist(rmfields(opt,'design','table','verbose','alpha'));

[p,t,stats,terms] = anovan(dat(:),design.anova,arg{:});

%% Provide output
if opt.verbose
  fprintf('-------\nResults\n-------\n')
  % Find col indices
  F = find(ismember(t(1,:),'F'));
  p = find(ismember(t(1,:),'Prob>F'));
  for ii=2:size(terms,1)+1
    if t{ii,p}<opt.alpha
      fprintf('''%s'' significant, F = %1.2f, p = %0.4f\n',...
        t{ii,1},t{ii,F},t{ii,p})
    else
      fprintf('''%s'' not significant, F = %1.2f, p = %0.4f\n',...
        t{ii,1},t{ii,F},t{ii,p})
    end
  end  
end
