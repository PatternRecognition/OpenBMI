function clab= getClabForLaplace(dat, tbf, laplace)
%clab= proc_laplace(dat, <filter_type>)
%clab= proc_laplace(dat, grid)
%clab= proc_laplace(dat, laplace)

if ~exist('laplace','var') | isempty(laplace), laplace= 'small'; end
if isstruct(laplace),
  if ~isfield(laplace, 'filter'), laplace.filter='small'; end
  if ischar(laplace.filter),
    laplace.filter= getLaplaceFilter(laplace.filter);
  end
else
  if iscell(laplace),
    grd= laplace;
    laplace= [];
    laplace.grid= grd;
    laplace.filter= [0 -1; 2 0; 0 1; -2 0]';
  else
    filter_type= laplace;
    laplace= [];
    laplace.grid= getGrid('grid_128');
    laplace.filter= getLaplaceFilter(filter_type);
  end
end

nChans= length(dat.clab);
pos= zeros(2, nChans);
for ic= 1:nChans,
  pos(:,ic)= getCoordinates(dat.clab{ic}, laplace.grid);
end

nRefs= size(laplace.filter,2);
tbf_idx= chanind(dat, tbf);
clab= dat.clab(tbf_idx);
for ic= tbf_idx,
  refChans= [];
  for ir= 1:nRefs,
    ri= find( pos(1,:)==pos(1,ic)+laplace.filter(1,ir) & ...
              pos(2,:)==pos(2,ic)+laplace.filter(2,ir) );
    refChans= [refChans ri];
  end
  if length(refChans)==nRefs,
    clab= cat(2, clab, dat.clab(refChans));
  end
end
clab= unique(clab);



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
 case 'small',
  filt= [0 -2; 2 0; 0 2; -2 0]';
 case 'large',
  filt= [0 -4; 4 0; 0 4; -4 0]';
 case 'horizontal',
  filt= [-2 0; 2 0]';
 case 'vertical',
  filt= [0 -2; 0 2]';
 case 'diagonal',
  filt= [-2 -2; 2 -2; 2 2; -2 2]';
 case 'eight',
  filt= [-2 0; -1 -1; 0 -2; 1 -1; 2 0; 1 1; 0 2; -1 1]';
 otherwise
  error('unknown filter matrix');
end
