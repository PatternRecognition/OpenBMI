function fv_fscore= proc_fisherScore(fv, varargin)
%PROC_FISHERSCORE - Computes the Fisher Score for each feature
%
%Synopsis:
% FV_SCORE= proc_fisherScore(FV, <OPT>)
%
%Arguments:
% FV  - data structure of feature vectors
% OPT - struct or property/value list of optional properties
%    .tolerate_nans: observations with NaN value are skipped
%       (nanmean/nanstd are used instead of mean/std)
%
%Returns:
% FV_SCORE - data structute of Fisher scores (one sample only)
%
%See also:
% proc_t_scaled, proc_r_values, proc_r_square, proc_rocAreaValues
%
%Comment:
% Only standard case is tested.

% Author(s): Benjamin Blankertz


if size(fv.y,1)>2,  %% multi-class: TODO! now doing it pairwise:
  warning('calculating pairwise Fisher scores');
  combs= nchoosek(1:size(fv.y,1), 2);
  for ic= 1:length(combs),
    ep= proc_selectClasses(fv, combs(ic,:));
    fv0= proc_fisherScore(ep);
    if ic==1,
      fv_fscore= fv0;
    else
      fv_fscore= proc_appendEpochs(fv_fscore, fv0);
    end
  end
  return; 
elseif size(fv.y,1)==1,
  warning('1 class only: calculating Fisher score against flat-line of same var');
  fv2= fv;
  szx= size(fv.x);
  fv2.x= fv2.x - repmat(mean(fv2.x,3), [1 1 size(fv2.x,3)]);
  fv2.className= {'flat'};
  fv= proc_appendEpochs(fv, fv2);
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'tolerate_nans',0);

sz= size(fv.x);
fv.x= reshape(fv.x, [prod(sz(1:end-1)), sz(end)]);
cl1= find(fv.y(1,:));
cl2= find(fv.y(2,:));
if opt.tolerate_nans,
  me= [nanmean(fv.x(:,cl1),2) nanmean(fv.x(:,cl2),2)];
  va= [nanstd(fv.x(:,cl1),0,2).^2 nanstd(fv.x(:,cl2),0,2).^2];
else
  me= [mean(fv.x(:,cl1),2) mean(fv.x(:,cl2),2)];
  va= [var(fv.x(:,cl1)')' var(fv.x(:,cl2)')'];
end
fscore= ((me(:,1)-me(:,2)).^2)./(va(:,1)+va(:,2)+eps);
fscore= reshape(fscore, [sz(1:end-1) 1]);

fv_fscore= copy_struct(fv, 'not', 'x','y','className');
fv_fscore.x= fscore;
if isfield(fv, 'className'),
  fv_fscore.className= {sprintf('Fs( %s , %s )', fv.className{1:2})};
end
fv_fscore.y= 1;
fv_fscore.yUnit= 'au';
