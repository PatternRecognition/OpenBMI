function dat= proc_variance(dat, nSections, calc_std)
%dat= proc_variance(dat, <nSections=1, calc_std=0>)
%
% calculate the variance in 'nSections' equally spaced intervals.
% works for cnt and epo structures.
%
% IN   dat       - data structure of continuous or epoched data
%      nSections - number of intervals in which var is to be calculated
%      calc_std  - standard deviation is calculated instead of variance
%
% OUT  dat       - updated data structure

% bb, ida.first.fhg.de

%% last changes
%% 14.02. 2005 stl : adapt .t field (values are set to the end of the
%%				     corresponding intervals)
%%

if ~exist('nSections','var'), nSections=1; end
if ischar(nSections) & strcmpi(nSections,'std'),
  nSections=1;
  calc_std=1;
end
if ~exist('calc_std','var') | (ischar(calc_std) & strcmpi(calc_std,'var')),
  calc_std= 0;
end
if ischar(calc_std) & strcmpi(nSections,'std'),
  calc_std=1;
end


 
[T, nChans, nMotos]= size(dat.x);
inter= round(linspace(1, T+1, nSections+1));
dat.t = [] ; 

xo= zeros(nSections, nChans, nMotos);
for s= 1:nSections,
  Ti= inter(s):inter(s+1)-1;
  if length(Ti)==1,
    warning('calculating variance of scalar');
  end
  if calc_std,
    xo(s,:,:)= reshape(std(dat.x(Ti,:),0,1), [1, nChans, nMotos]);
  else
    if length(Ti)==1,
      xo(s,:,:)= dat.x(Ti,:);
    else
      if nChans*nMotos*length(Ti)<=10^6;
        xo(s,:,:)= reshape(var(dat.x(Ti,:)), [1, nChans, nMotos]);
      else
        for i=1:nMotos
          xo(s,:,i) = var(dat.x(Ti,:,i));
        end
      end
    end
  end
  dat.t(s) = Ti(end);
end

dat.x= xo;
