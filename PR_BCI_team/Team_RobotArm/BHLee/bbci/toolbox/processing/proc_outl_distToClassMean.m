function [fv, outlier_idx, dist]= ...
    proc_outl_distToClassMean(fv, thresh, varargin)
% [fv, outl_idx] = proc_outl_distToClassMean(fv, thresh, <opt>)
% 
% Mark or remove samples with large distance to their respective class
% mean.
%
% IN  fv     - struct of feature vectors
%     thresh - threshold, the interpretation depends on opt.policy
%     opt    - struct and/or property/value list with optional parameters
%     .policy   - in the following policies, outliers are defined as...
%                 'abs_dist': samples with larger absolute distance than
%                             'thresh', 
%                 'perc_of_dist' (default): the largest 'thresh' percent of
%                             the sample distances, 
%                 'perc_of_gauss': the top 'thresh' gauss percentile of the
%                             sample distances, 
%                 'perc_of_samples': the top 'thresh' percent of the number
%                             of samples.
%     .distance - the used distance function. can be 'euclidean' or
%                 'mahalanobis' (default).
%     .remove_outliers - if this flag is set to 1 (default), trials that 
%                 are detected as outliers are removed from the structure fv,
%                 otherwise only their label is set to 0.
%     .display  - if true: generate a plot with variances, default: 0.
%
% OUT:
% fv       - updated epoched data struct.
% outl_idx - indices of the detected outliers

% kraulem, blanker 03-07/2004


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'policy', 'perc_of_dist', ...
                  'distance', 'mahalanobis', ...
                  'display', 0, ...
                  'remove_outliers', 1, ...
                  'handle',[]);

if isempty(opt.handle) & opt.display
  opt.handle = figure;
end


xx= getfield(proc_flaten(fv), 'x');

% calculate the distance of each sample to its class mean
%    from the mean.
%me1 = mean(xx,2);
switch(lower(opt.distance)), 
 case 'mahalanobis',
  for i = 1:size(fv.y,1)
    x_cl = xx(:,find(fv.y(i,:)));
    me1 = mean(x_cl,2);
    coxx = cov(x_cl');
    coxx = pinv(coxx);
    [R,p] = chol(coxx);
    if p~=0
      % coxx not positive definite.
      coxx = cov(x_cl')+eye(size(fv.x,1));
      coxx = pinv(coxx); 
    end
    dist(find(fv.y(i,:))) = mahalanobis_dist(me1,x_cl,coxx);
  end
 case 'euclidean',
  for i = 1:size(fv.y,1)
    x_cl = xx(:,find(fv.y(i,:)));
    me1 = mean(x_cl,2);
    dist(find(fv.y(i,:))) = mahalanobis_dist(me1,x_cl);
  end
 otherwise,
  error('distance measure not known');
end


if strcmpi(opt.policy, 'perc_of_gauss'),
  me = mean(dist);
  st = var(dist);
  thresh = sqrt(st)*erfinv(1-2*thresh/100)+me;
  opt.policy= 'abs_dist';
end

nSamples= length(dist);
switch(lower(opt.policy)),
 case 'abs_dist',
  outlier_idx = find(dist>thresh);
 case 'perc_of_samples',
  [so,si] = sort(dist);
  outlier_idx = si(end-round(thresh/100*nSamples)+1:end);
 case 'perc_of_dist',
  [so,si] = sort(dist);
  perc= 100-100*cumsum(so/sum(so));
  iCut= min([find(perc<thresh) nSamples]);
  outlier_idx = si(iCut+1:end);
 otherwise,
  error('policy not known');
end

if opt.remove_outliers,
  fv= proc_selectEpochs(fv, 'not', outlier_idx);
else
  fv.y(:,outlier_idx)= NaN;
end
  

if opt.display
  figure(opt.handle);
  [so,si1] = sort(dist);
  [dum,si2] = sort(si1);
  reg_idx= setdiff(1:nSamples, outlier_idx);
  plot(si2(reg_idx), dist(reg_idx), '.g');
  hold on;
  plot(si2(outlier_idx), dist(outlier_idx), '.r');
  hold off;
  title('sorted sample to class mean distances');
  xlabel('sorted samples');
  ylabel(sprintf('distances [%s] to class mean', opt.distance));
end
