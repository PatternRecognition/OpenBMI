function fv_p= proc_alpha_steps(fv, alphas)
%PROC_ALPHA_STEPS - Determine what levels of significant are reached
%
%Description:
%  This function t-scales the data (Student's t-test), calculates
%  the threshold for specified alpha levels, and returns step-function
%  time series, which indicate how many threshold are surpassed.
%  Only for two-class data.
%
%  This function is intended to give input to the function
%  grid_addBars of the BBCI toolbox.
%
%Usage:
%  FV_A= proc_alpha_steps(FV, <ALPHAS=[0.1 0.05 0.01 0.001]>)
%
%Input:
%  FV:     Data structure of feature vectors
%  ALPHAS: Levels of significance
%
%Output:
%  FV_A:   Data structure of alpha-step functions. A value of 2 means that
%          to levels of significance were surpassed at that point.
%
%See also proc_t_scale

%% blanker@first.fhg.de, 01/2005


if nargin<2,
  alphas= [0.1 0.05 0.01 0.001];
end
alphas= -sort(-alphas);

fv_tsc= proc_t_scale(fv);
levels= calcTcrit(alphas, fv_tsc.df);

fv_p= copy_struct(fv_tsc, 'not','x');
fv_p.crit= levels;
fv_p.alpha= alphas;
fv_p.yUnit= 'au';
fv_p.className= strrep(fv_p.className, 't-scaled', 'alpha-steps');

idx= 1:size(fv_tsc.x,1);
if isfield(fv, 'refIval'),
  ref_idx= getIvalIndices(fv.refIval, fv);
  idx= setdiff(idx, ref_idx);
end

fv_p.x= zeros(size(fv_tsc.x));
for ll= 1:length(levels),
  fv_p.x(idx,:)= fv_p.x(idx,:) + [abs(fv_tsc.x(idx,:))>levels(ll)];
end
