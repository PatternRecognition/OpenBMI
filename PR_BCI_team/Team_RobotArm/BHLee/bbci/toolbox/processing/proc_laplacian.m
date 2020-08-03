
function [varargout]= proc_laplacian(dat, varargin)
%PROC_LAPLACIAN - Apply spatial Laplacian filter to signals
%
%Synopsis:
% [DAT_LAP, LAP_W]= proc_laplacian(DAT, <OPT>)
%
%Arguments:
% DAT: data structure of continuous or epoched data
% OPT: struct or proerty/value list of optional properties:
%  .filter_type    - {small, large, horizontal, vertical, diagonal, eight},
%                     default 'small'
%  .clab - channels that are to be obtained, default '*'.
%  .ignore_clab: labels of channels to be ignored, default {'E*'}.
%  .require_complete_neighborhood
%  .verbose
%
%Returns:
%  DAT_LAP: updated data structure
%  LAP_W:   filter matrix that can be used, e.g. in proc_linearDerivation
%
%See also:
%  getClabForLaplacian, proc_linearDerivation

% blanker@cs.tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
  'clab', '*', ...
  'ignore_clab', {'E*'}, ...
  'copy_clab', {'E*'}, ...
  'grid', 'grid_128', ...
  'filter_type', 'small', ...
  'require_complete_neighborhood', 1, ...
  'require_complete_output', 0, ...
  'appendix', ' lap', ...
  'verbose', 0);

if ~iscell(opt.ignore_clab),
  opt.ignore_clab= {opt.ignore_clab};
end

laplace= [];
laplace.grid= getGrid(opt.grid);
if isnumeric(opt.filter_type),
  laplace.filter= opt.filter_type;
elseif isequal(opt.filter_type, 'flexi')
  laplace.filter= [];
  % for standard positions C3, CP5, etc
  laplace.filter1= getLaplaceFilter('small');
  % for extended positions CFC3, PCP1, etc
  laplace.filter2= getLaplaceFilter('diagonal_small');
else
  laplace.filter= getLaplaceFilter(opt.filter_type);
end

if ~isstruct(dat),
  dat= struct('clab',{dat});
end

rc= chanind(dat, {'not', opt.ignore_clab{:}});
nOrigChans= length(dat.clab);
pos= zeros(2, nOrigChans);
for ic= 1:nOrigChans,
  pos(:,ic)= getCoordinates(dat.clab{ic}, laplace.grid);
end
pos(:,setdiff(1:nOrigChans,rc))= inf;

idx_tbf= chanind(dat, opt.clab);
W= zeros(length(dat.clab), length(idx_tbf));
clab = [];
lc= 0;
filter_tmp = laplace.filter;
for ci= 1:length(idx_tbf),
  cc= idx_tbf(ci);
  refChans= [];
  if isequal(opt.filter_type, 'flexi'),
    clab_tmp= strrep(dat.clab{cc}, 'z','0');
    if sum(isletter(clab_tmp))<3,
      laplace.filter= laplace.filter1;
    else
      laplace.filter= laplace.filter2;
    end
  end
  if isequal(opt.filter_type, 'eleven'),
    if isnan(mod(str2double(dat.clab{cc}(end)),2))
      warning('asymmetric filter type ''eleven'' ignores central channels');
      continue;
    end
  end
  if size(filter_tmp,3) > 1
    if isequal(dat.clab{cc}(end),'z')
      laplace.filter = filter_tmp(:,:,2);
    elseif mod(str2double(dat.clab{cc}(end)),2)
      laplace.filter = filter_tmp(:,:,1);
    else
      laplace.filter = filter_tmp(:,:,end);
    end
  end
  nRefs= size(laplace.filter,2);
  for ir= 1:nRefs,
    ri= find( pos(1,:)==pos(1,cc)+laplace.filter(1,ir) & ...
      pos(2,:)==pos(2,cc)+laplace.filter(2,ir) );
    refChans= [refChans ri];
  end
  if length(refChans)==nRefs | ~opt.require_complete_neighborhood,
    lc= lc+1;
    W(cc,lc)= 1;
    if ~isempty(refChans),
      W(refChans,lc)= -1/length(refChans);
    end
    clab= [clab, dat.clab(cc)];
    if opt.verbose,
      fprintf('%s: ref''ed to: %s\n', ...
        dat.clab{cc}, vec2str(dat.clab(refChans)));
    end
  elseif opt.require_complete_output,
    error('channel %s has incomplete neighborhood', dat.clab{cc});
  end
end
clear filter_tmp
W= W(:,1:lc);

if isfield(dat, 'x'),
  out= proc_linearDerivation(dat, W, 'clab', strcat(clab, opt.appendix));
  if ~isempty(opt.copy_clab),
    out= proc_copyChannels(out, dat, opt.copy_clab);
  end
  varargout= {out, W};
else
  varargout= {W};
end



function pos= getCoordinates(lab, grid)

nRows= size(grid,1);
%w_cm= warning('query', 'bci:missing_channels');
%warning('off', 'bci:missing_channels');
ii= chanind(grid, lab);
%warning(w_cm);
if isempty(ii),
  pos= [NaN; NaN];
else
  xc= 1+floor((ii-1)/nRows);
  yc= ii-(xc-1)*nRows;
  xc= 2*xc - isequal(grid{yc,1},'<');
  pos= [xc; yc];
end



function filt= getLaplaceFilter(filter_type)

switch lower(filter_type),
  case 'sixnew'
    filt = [-4 0; -2 0; 0 -2; 0 2; 2 0; 4 0]';
  case 'eightnew'    
    filt = [-4 0; -2 -2; -2 0; -2 2; 2 -2; 2 0; 2 2; 4 0]';
  case 'small',
    filt= [0 -2; 2 0; 0 2; -2 0]';
  case 'large',
    filt(:,:,1) = [-2 0; 0 -2; 0 2; 2 0; 4 0; 8 0]';
    filt(:,:,2) = [-4 0; -2 0; 0 -2; 0 2; 2 0; 4 0]';
    filt(:,:,3) = [-8 0; -4 0; -2 0; 0 -2; 0 2; 2 0]';  
  case 'horizontal',
    filt= [-2 0; 2 0]';
  case 'vertical',
    filt= [0 -2; 0 2]';
  case 'bip_to_anterior';
    filt= [0 -2]';
  case 'bip_to_posterior';
    filt= [0 2]';
  case 'bip_to_left';
    filt= [-2 0]';
  case 'bip_to_right';
    filt= [2 0]';
  case 'diagonal',
    filt= [-2 -2; 2 -2; 2 2; -2 2]';
  case 'diagonal_small',
    filt= [-1 -1; 1 -1; 1 1; -1 1]';
  case 'six',
    filt= [-2 0; -1 -1; 1 -1; 2 0; 1 1; -1 1]';
  case 'eightsparse',
    filt= [-2 0; -2 -2; 0 -2; 2 -2; 2 0; 2 2; 0 2; -2 2]';
  case 'eight',
    filt= [-2 0; -1 -1; 0 -2; 1 -1; 2 0; 1 1; 0 2; -1 1]';
  case 'ten'
    filt= [-4 0; -2 -2; -2 0; -2 2; 0 -2; 0 2; 2 -2; 2 0; 2 2; 4 0]';
  case 'eleven_to_anterior'
    % eleven unsymmetric neighbors for channel in the left emisphere
    % (neigbors more going to the left)
    filt(:,:,1) = [-4 0; -4 2; -2 -2; -2 0; -2 2; -2 4; 0 -2; 0 2; 0 4; 2 0; 2 2]';
    % eleven unsymmetric neighbors for channel in the right emisphere
    % (neigbors more going to the right)
    filt(:,:,2) = [-2 0; -2 2; 0 -2; 0 2; 0 4; 2 -2; 2 0; 2 2; 2 4; 4 0; 4 2]';
  case 'eleven'
    filt(:,:,1) = [-4 -2; -4 0; -4 2; -2 -2; -2 0; -2 2; 0 -2; 0 2; 2 -2; 2 0; 2 2]';
    filt(:,:,2) = [-2 -2; -2 0; -2 2; 0 -2; 0 2; 2 -2; 2 0; 2 2; 4 -2; 4 0; 4 2]';
  case 'twelve'
    filt = [-2 0; -2 -2; 0 -2; 2 -2; 2 0; 2 2; 0 2; -2 2; -1 -1; 1 -1; 1 1; -1 1]';  
  case 'eighteen',
    filt= [-2 2; 0 2; 2 2; -3 1; -1 1; 1 1; 3 1; -4 0; -2 0; 2 0; 4 0; -3 -1; -1 -1; 1 -1; 3 -1; -2 -2; 0 -2; 2 -2]';
  case 'twentytwo'
    filt = [-1 3; 1 3; -2 2; 0 2; 2 2; -3 1; -1 1; 1 1; 3 1; -4 0; -2 0; 2 0; 4 0; -3 -1; -1 -1; 1 -1; 3 -1; -2 -2; 0 -2; 2 -2; -1 -3; 1 -3]';
  otherwise
    error('unknown filter matrix');
end

