function [fv, outlier_idx, dist1, dist2] = proc_outl_var1(fv,varargin)
% [fv, outl_idx, dist_trial, dist_channel] = proc_outl_var(fv,<opt>)
% 
% Find and remove channels and trials with high variances in epoched data.
% IN:
% opt - struct with optional parameters.
% fv  - feature vectors for Xvalidation.
% possible fields:
%     .display  - if true: generate a plot with variances. default: true 
%     .chanthresh - threshold for discarding channels (concerns dist of variance)
%                 if chanthresh == 0: do not discard <default>
%     .trialthresh - threshold for discarding trials (concerns dist of variance)
%                 if trialthresh == 0: do not discard <default>
%     .selectClasses - variance will only be calculated for these channels.
%                     nevertheless, fv will be given back with all channels.
%                    <default 'all'>.
%     .distance   - the used distance function. can be 'euclidean' or
%                   'mahalanobis' (default).
%     .trialperc - to remove trialperc percent trials.
%     .trialgauss - to remove trialgauss percent of an estimated gauss percentile.
%     .remove_outliers - if this flag is set to 1 (default), trials that 
%                 are detected as outliers are removed from the structure fv,
%                 otherwise only their label is set to 0.
%     .classwise - distances are calculated for each class separately. 
%                  default 0.
%
% OUT:
% fv       - updated epoched data struct.
% outl_idx - indices of the detected outliers
% dist_trial - distance from class mean for each trial (before removing)
% dist_channel - distance from class mean for each channel (before removing)
%
% kraulem 09/03/2004

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'display', true,...
                  'chanthresh', 0,...
                  'trialthresh', 0,...
                  'trialperc',0,...
                  'trialgauss',0,...
                  'remove_outliers',1,...
                  'selectClasses', 'all',...
                  'distance', 'mahalanobis',...
                  'handles',[],...
		  'classwise',0);
fig_opt = {'numberTitle','off',...
	   'menuBar','none'};
if opt.trialthresh(1)*opt.trialperc(1)>0
  error('Only one value of trialtresh and trialperc is allowed to be bigger than zero');
end

if strcmp(opt.selectClasses,'all')
  ev = size(fv.x,3);
  vari = proc_variance(fv);
  nClasses = size(fv.y,1);
else 
  [fv2,ev] = proc_selectClasses(fv,opt.selectClasses);
  vari = proc_variance(fv2);
  nClasses = size(fv2.y,1);
end
vari_all = squeeze(vari.x);
vari_all = log(vari_all);

if opt.classwise
  % variances are to be calculated for each class separately.
  varianceNum = nClasses;
else
  % all classes are to be put together.
  varianceNum = 1;
end
outlier_idx = [];
if opt.display & isempty(opt.handles)
  opt.handles(1) = figure;
end
for var_ind = 1:varianceNum
  if length(opt.handles)<2*var_ind & opt.display
    opt.handles(2*var_ind-1) = figure(opt.handles(end)+1);
    opt.handles(2*var_ind) = figure(opt.handles(end)+1);
  end

  
  if opt.classwise
    % select only the necessary class.
    ev_cl{var_ind} = find(fv.y(var_ind,:));
    for i = 2:nClasses
      if length(opt.trialthresh)<nClasses
	opt.trialthresh(i) = opt.trialthresh(1);
      end
      if length(opt.trialperc)<nClasses
	opt.trialperc(i) = opt.trialperc(1);
      end
    end
  else
    % take all classes
    ev_cl{var_ind} = 1:size(fv.y,2);
  end
  vari = vari_all(:,ev_cl{var_ind});
  
  % calculate the mahalanobis distance of each variance of each trial
  %    from the mean.  
  me1 = mean(vari,2);
  if strcmp(opt.distance,'mahalanobis')
    covari = cov(vari');
    if rank(covari)<size(covari,1)
      warning('Bandpowers are highly correlated!');
      [V,D] = eig(covari);
      D = diag(D);
      % only keep eigenvalues above a threshold:
      D_ind = find(D>size(covari,1)*norm(covari)*1e-8);
      covari = V(:,D_ind)*diag(1./D(D_ind))*V(:,D_ind)';
    else
      covari = pinv(covari);
    end
    dist1 = mahalanobis_dist(me1,vari,covari);
  elseif strcmp(opt.distance,'euclidean')
    dist1 = mahalanobis_dist(me1,vari);
  end
  % calculate the mahalanobis distance of the average variance of each channel
  %    from the mean.
  %covari = cov(vari);
  %covari = pinv(covari);
  %me2 = mean(vari,1)';
  % switch(opt.distance),
  %    case 'mahalanobis',
  %     dist2 = sqrt(mahalanobis_dist(me2,vari',covari));
  %    case 'euclidean',
  %     dist2 =  sqrt(mahalanobis_dist(me2,vari'));
  %    otherwise
  %     error('unknown distance method');
  %   end
  dist2 = abs(me1-mean(me1));

  % remove channels:
  if opt.chanthresh>0
    ind = find(dist2>opt.chanthresh);
    if isfield(fv,'clab') & isfield(fv,'x')
      fv.clab(ind) = [];
      fv.x(:,ind,:) = [];
    end
    vari(ind,:) = [];
  end

  if opt.trialgauss>0
    me = mean(dist1);
    st = var(dist1);
    opt.trialthresh = sqrt(st)*erfinv(1-2*opt.trialgauss)+me;
  end

  if opt.trialthresh(var_ind)>0
    % mark outliers due to trialthresh criterium
    outlier_idx_tmp = find(dist1>opt.trialthresh(var_ind));
    vari(:,outlier_idx_tmp) = [];
    outlier_idx = [outlier_idx, ev_cl{var_ind}(outlier_idx_tmp)];
  elseif opt.trialperc(var_ind)>0
    % mark outliers due to trialperc criterium
    [dum,ind] = sort(dist1);
    outlier_idx = ind(end-round(opt.trialperc*length(dist1))+1:end);
    vari(:,outlier_idx) = [];
  else 
    outlier_idx = [];
  end
   
  if opt.display
    % generate plots.
    %subplot(2,1,1);
    figure(opt.handles(2*var_ind-1))
    [m,ind] = sort(dist1);
    plot(m);
    hold on;
    l = line(get(gca,'XLim'),[1 1]*opt.trialthresh(var_ind));
    set(l,'Color',[1 0 0]);
    set(get(gca,'Title'),'String','trial variance distances');
    if opt.classwise
      name_str = ['Distances for class "' fv.className{var_ind} '"'];
    else 
      name_str = ['Distances for all classes'];
    end
    set(gcf,fig_opt{:},'name',name_str);
    hold off;
    %subplot(2,1,2);
    figure(opt.handles(2*var_ind))
    imagesc(vari);
    % plot(m);
    if opt.classwise
      name_str = ['Variances for class "' fv.className{var_ind} '"'];
    else 
      name_str = ['Variances for all classes'];
    end
    set(gcf,fig_opt{:},'name',name_str);
    if opt.trialthresh(var_ind)>0
      set(get(gca,'Title'),'String','variance distances, outlier trials already removed.');
    else
      set(get(gca,'Title'),'String','variance distances');
    end
    set(gca,'YTick',1:size(fv.x,2));
    set(gca,'YTickLabel',fv.clab);
    xlabel('Trials');
    ylabel('Channels');
    colorbar;
  end
end
 
if opt.remove_outliers,
  fv= proc_removeEpochs(fv, outlier_idx);
else
  fv.y(:,outlier_idx)= NaN;
end
