function fv = proc_outl_slow(fv,varargin)
% fv = proc_outl_slow(fv,<opt>)
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
%
% OUT:
% fv - updated epoched data struct.
%

% kraulem 09/03/2004

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'display', true,...
                  'chanthresh', -1,...
                  'trialthresh', 0,...
                  'selectClasses', 'all',...
                  'distance', 'mahalanobis',...
                  'handles',[]);
if isempty(opt.handles)
  opt.handles(1) = figure;
  opt.handles(2) = figure;
end
if strcmp(opt.selectClasses,'all')
  vari = fv;
else 
  vari = proc_selectClasses(fv,opt.selectClasses);
end
vari = vari.x;

[di1,di2,di3] = size(vari);
% calculate the mahalanobis distance of each values of each trial
%    from the mean.
vari1 = reshape(vari,[di1*di2,di3]);
covari = cov(vari1');
covari = pinv(covari);
me1 = mean(vari1,2);
if strcmp(opt.distance,'mahalanobis')
  dist1 = mahalanobis_dist(me1,vari1,covari);
elseif strcmp(opt.distance,'euclidean')
  dist1 = mahalanobis_dist(me1,vari1);
end


% calculate channel outliers:
if opt.chanthresh>-1

  % calculate the mahalanobis distance of each channel
  %    from the mean.
  vari2 = reshape(permute(vari,[2 1 3]),[di2,di1*di3]);
  covari = cov(vari2);
  covari = pinv(covari);
  me2 = mean(vari2,1)';
  if strcmp(opt.distance,'mahalanobis')
    dist2 = mahalanobis_dist(me2,vari2',covari);
  elseif strcmp(opt.distance, 'euclidean')
    dist2 =  mahalanobis_dist(me2,vari2');
  end
  
  % remove channels:
  if opt.chanthresh>0
    ind = find(dist2>opt.chanthresh);
    if isfield(fv,'clab') & isfield(fv,'x')
      fv.clab(ind) = [];
      fv.x(:,ind,:) = [];
    end
    vari(:,ind,:) = [];
    vari2(ind,:) = [];
  end
end

% remove trials:
if opt.trialthresh>0
  ind = find(dist1>opt.trialthresh);
  if isfield(fv,'x')&isfield(fv,'y')
    fv.x(:,:,ind) = [];
    fv.y(:,ind) = [];
  end
  if isfield(fv,'latency')
    fv.latency(ind) = [];
  end
  if isfield(fv,'bidx')
    fv.bidx(ind) = [];
  end
  
  vari(:,:,ind) = [];
end

if opt.display
  % generate plots.
  %subplot(2,1,1);
  figure(opt.handles(1))
  [m,ind] = sort(dist1);
  plot(m);
  hold on;
  l = line(get(gca,'XLim'),[1 1]*opt.trialthresh);
  set(l,'Color',[1 0 0]);
  set(get(gca,'Title'),'String','trial variance distances');
  hold off;
  
  if opt.chanthresh>-1
    figure(opt.handles(2))
    imagesc(vari2);
    % plot(m);
    if opt.trialthresh>0
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
