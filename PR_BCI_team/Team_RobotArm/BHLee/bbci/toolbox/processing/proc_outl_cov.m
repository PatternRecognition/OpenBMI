function [fv, outlier_idx, dist]= proc_outl_cov(fv, varargin);

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'classwise', 1, ...
                  'percentiles', [20 80], ...
                  'threshold_factor', 1, ...
                  'remove_outliers', 1, ...
                  'iterative', 0, ...
                  'display', 0);

[T, nChans, nTrials]= size(fv.x);
fv_cov= proc_covariance(fv);
fv_cov.x= reshape(fv_cov.x, [nChans nChans nTrials]);

if opt.iterative,
  goon= 1;
  hist= zeros(nTrials, 1);
  while goon,
    [dist, th]= outl_cov(fv_cov, opt);
    if exist('thresh','var'),  %% fix threshold in the first pass
      dist= outl_cov(fv_cov, opt);
    else
      [dist, thresh, perc]= outl_cov(fv_cov, opt);
    end
    [maxdist,mi]= max(dist);
    if maxdist > thresh,
      fv_cov.y(:,mi)= NaN;
      hist(mi)= dist(mi);
    else
      goon= 0;
    end
  end
  outlier_idx= find(any(isnan(fv_cov.y),1));
  dist(outlier_idx)= hist(outlier_idx);
else
  [dist, thresh, perc]= outl_cov(fv_cov, opt);
  outlier_idx= find(dist > thresh);
end

if opt.display,
%  if opt.iterative,
%    warning('display in iterative mode is questionable');
%  end
  clf; 
  boxplot(dist, 'notch',1, ...
          'whiskerpercentiles',opt.percentiles, ...
          'whiskerlength',opt.threshold_factor)
  hold on;
  plot(linspace(1.15, 1.45, length(dist)), sort(dist));
  set(gca, 'XLim', [0.9 1.5], 'XTick',[]);
  line([1.15 1.15; 1.45 1.45], [1;1]*perc, 'Color','k', 'LineStyle','--');
  line([1.15 1.45], [1 1]*thresh, 'Color','r');
  hold off; 
  xlabel('');
end

if opt.remove_outliers,
  fv= proc_removeEpochs(fv, outlier_idx);
else
  fv.y(:,outlier_idx)= NaN;
end



%% -----


function [dist, thresh, perc]= outl_cov(fv_cov, opt);

nTrials= size(fv_cov.x, 3);
dist= NaN * ones(nTrials, 1);
for tr= 1:nTrials,
  label= [1 2] * fv_cov.y(:,tr);
  if ismember(label, [1 2]),
    if opt.classwise,
      idx= setdiff(find(fv_cov.y(label,:)), tr);
    else
      idx= setdiff(1:nTrials, tr);
    end
    S= mean(fv_cov.x(:,:,idx), 3);
    dist(tr)= trace(inv(S)*fv_cov.x(:,:,tr));
  end
end

perc= percentiles(dist, opt.percentiles);
thresh= perc(2) + diff(perc) * opt.threshold_factor;
