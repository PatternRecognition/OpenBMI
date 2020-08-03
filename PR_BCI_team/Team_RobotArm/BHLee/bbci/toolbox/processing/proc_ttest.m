function fv= proc_ttest(fv, varargin)
%PROC_TTEST - Perform a pointwise t-test
%
%FV= proc_ttest(FV, <OPT>)
%
% IN  FV     - data structure of feature vectors
%     OPT    - struct or property/value list of optional properties:
%     'alpha' - significance level, default 0.01
%     'tail'  - can be 'both' (default), 'right', 'left', see ttest2
%     'vartype' - can be 'equal' (default) or 'unequal', see ttest2
%
% OUT FV     - data structute of t-scaled data (one class only)
%            .crit - threshold of 'significance' of two-sided t-test
%                    with respect to level alpha
%
% Note that FV.h cannot be interpreted as significance
% as observations at consecutive time points and in different
% channels are (usually) not independent observations (-> problem
% of multiple testing).
% To display the *.crit thresholds in grid plots use the function
% grid_markRange.
%
% SEE proc_r_square, grid_plot, grid_markRange


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'alpha', 0.01, ...
                  'tail', 'both', ...
                  'vartype', 'equal');

nClasses= size(fv.y, 1);
if nClasses>2,
  error;
end

if nClasses==1,
  [fv.h, fv.p, fv.ci, fv.stats]= ...
      ttest(fv.x, 0, opt.alpha, opt.tail, 3);
  fv.className= {sprintf('%s [t-scaled]', fv.className{1})};
else
  c1= find(fv.y(1,:));
  c2= find(fv.y(2,:));
  [fv.h, fv.p, fv.ci, fv.stats]= ...
      ttest2(fv.x(:,:,c1), fv.x(:,:,c2), opt.alpha, opt.tail, opt.vartype, 3);
  fv.className= {sprintf('%s - %s [t-scaled]', fv.className{1:2})};
end

fv.x= fv.stats.tstat;
fv.alpha= opt.alpha;
fv.y= 1;
fv.yUnit= 't';
