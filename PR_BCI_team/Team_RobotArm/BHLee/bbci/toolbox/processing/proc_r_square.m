function fv= proc_r_square(fv, varargin)
%fv_rsqu= proc_r_square(fv, <opt>)
%
% Computes the r^2 value for each feature. The r^2 value is a measure
% of how much variance of the joint distribution can be explained by
% class membership.
% Any suggestion for a good reference?
%
% IN   fv  - data structure of feature vectors
%      opt - struct or property/value list of optional properties
%       .tolerate_nans: observations with NaN value are skipped
%            (nanmean/nanstd are used instead of mean/std)
%
% OUT  fv_rsqu - data structute of r^2 values (one sample only)
%
% SEE  proc_t_scaled, proc_r_values

% bb 03/2003, ida.first.fhg.de


if length(varargin)>0 & isstruct(varargin{1}),
  fv2= varargin{1};
  varargin= varargin(2:end);
  if size(fv.y,1)*size(fv2.y,1)>1,
    error('when using 2 data sets both may only contain 1 sigle class');
  end
  if strcmp(fv.className{1}, fv2.className{1}),
    fv2.className{1}= strcat(fv2.className{1}, '2');
  end
  fv= proc_appendEpochs(fv, fv2);
  clear fv2;
end

fv= proc_r_values(fv, varargin{:});
fv.x= fv.x.^2;
for cc= 1:length(fv.className),
  fv.className{cc}= ['r^2' fv.className{cc}(2:end)];
end
fv.yUnit= 'r^2';



return


%% function as adapted from what Gerwin Schalk supplied
%% (variance is normalized by N not N-1)
%
%c1= find(fv.y(1,:));
%c2= find(fv.y(2,:));
%lp= length(c1);
%lq= length(c2);
%sz= size(fv.x);
%rsqu= zeros(sz(1:2));
%for ti= 1:sz(1),
%  for ci= 1:sz(2),
%    p= fv.x(ti,ci,c1);
%    q= fv.x(ti,ci,c2);
%    sp= sum(p, 3);
%    sq= sum(q, 3);
%    g= (sp+sq)^2 / (lp+lq);
%    rsqu(ti, ci)= ( sp^2/lp + sq^2/lq - g ) / ( sum(p.^2) + sum(q.^2) - g );
%  end
%end
