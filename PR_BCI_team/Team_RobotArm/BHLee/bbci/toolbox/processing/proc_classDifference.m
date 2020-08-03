function out= proc_classDifference(epo, clInd, weight)
%epo= proc_classDifference(epo, clInd, <weighted=0>)
%epo= proc_classDifference(epo, epo2, <weighted=0>)
%
% calculate the difference between the class means (clInd(1)-clInd(2)).
% the difference is weighted according to the size of the classes,
% if argument 'weighted' is set to 1.
%
% IN   epo      - data structure of epoched data
%      epo2     - data structure of epoched data to be subtracted
%      clInd    - classes of which the difference is to be calculated:
%                 names of classes (strings in a cell array), or 
%                 vector of class indices
%      weighted - if 1 class means are weighted according to their size
%                 (usually this does not make sense)
%
% OUT  epo      - updated data structure

% bb, ida.first.fhg.de

if ~exist('clInd', 'var') | isempty(clInd), clInd= [1 2]; end
if ~exist('weight', 'var'), weight=0; end

if isstruct(clInd),          %% clInd is epo2
  erp1= proc_average(epo);
  erp2= proc_average(clInd);
elseif iscell(clInd),
  erp1= proc_average(epo, clInd{1});
  erp2= proc_average(epo, clInd{2});
else
  erp1= proc_average(epo, epo.className{clInd(1)});
  erp2= proc_average(epo, epo.className{clInd(2)});
end

out= copyStruct(epo, 'x','y','N','className');
out.N= [erp1.N erp2.N];
if weight,
  out.x= ( erp1.x*erp1.N - erp2.x*erp2.N ) / sum(out.N);
else
  out.x= erp1.x - erp2.x;
end
out.y= 1;
out.className= {sprintf('%s - %s', erp1.className{1}, erp2.className{1})};
