function fv_tsc= proc_t_scale(fv, alpha)
%fv_tsc= proc_t_scale(fv, <alpha=0.01>)
%
% calculates the pointwise t-scaled difference between events
% of two classes.
%
% IN  fv     - data structure of feature vectors
%     alpha  - significance level
%
% OUT fv_tsc       - data structute of t-scaled data (one sample only)
%            .crit - threshold of 'significance' of two-sided t-test
%                    with respect to level alpha
%
% note that fv_tsc.crit cannot be interpreted as significance
% as observations at consecutive time points and in different
% channels are (usually) not independent observations (-> problem
% of multiple testing).
% nevertheless the t-scaled difference provides some impression of
% how much of the difference of the class averages are only due to
% 'outliers'.
% to display the *.crit thresholds in grid plots use the function
% grid_markRange
%
% SEE proc_r_square, grid_plot, grid_markRange

% bb 03/03, ida.first.fhg.de


if ~exist('alpha', 'var') | isempty(alpha), alpha= 0.01; end

nClasses= size(fv.y, 1);
if nClasses==1,
  bbci_warning('1 class only: calculating r-values against flat-line of same var', 'r_policy');
  fv2= fv;
  szx= size(fv.x);
  fv2.x= fv2.x - repmat(mean(fv2.x,3), [1 1 size(fv2.x,3)]);
  fv2.className= {'flat'};
  fv= proc_appendEpochs(fv, fv2);
elseif nClasses>2,
  warning('calculating pairwise t-scaled values');
  state= bbci_warning('off', 'selection');
  combs= fliplr(nchoosek(1:size(fv.y,1), 2));
  for ic= 1:length(combs),
    ep= proc_selectClasses(fv, combs(ic,:));
    if ic==1,
      fv_tsc= proc_t_scale(ep,alpha);
% begin sthf       
      ndims = length(size(ep.x));
      if ndims > 3
        fv_tsc.ndims = ndims;
      end
% end sthf 
    else
      fv_tsc= proc_appendEpochs(fv_tsc, proc_t_scale(ep,alpha));
    end
  end
  bbci_warning(state);
  return; 
end

sz= size(fv.x);
fv.x= reshape(fv.x, [prod(sz(1:end-1)) sz(end)]);

c1= find(fv.y(1,:));
c2= find(fv.y(2,:));
N1= length(c1);
N2= length(c2);
df= N1+N2-2;
sxd= sqrt( ((N1-1)*var(fv.x(:,c1)')'+(N2-1)*var(fv.x(:,c2)')') / df  * ...
           (1/N1+1/N2) );
x_tsc= (mean(fv.x(:,c1),2)-mean(fv.x(:,c2),2)) ./ sxd;
x_tsc= reshape(x_tsc, sz(1:end-1));

dont_copy= {'x','y','className'};
if isfield(fv, 'indexedByEpochs'),
  dont_copy= cat(2, dont_copy, {'indexedByEpochs'}, fv.indexedByEpochs);
end

fv_tsc= copy_struct(fv, 'not', dont_copy{:});
fv_tsc.x= x_tsc;
fv_tsc.df= df;
fv_tsc.crit= calcTcrit(alpha, df);
fv_tsc.alpha= alpha;
fv_tsc.y= 1;
fv_tsc.className= {sprintf('%s - %s [t-scaled]', fv.className{1:2})};
fv_tsc.yUnit= 't';
