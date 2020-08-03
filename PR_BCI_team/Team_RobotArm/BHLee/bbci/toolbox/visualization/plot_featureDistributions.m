function h = plot_featureDistributions(fv,varargin)
% h = plot_featureDistributions(fv,<h,hyperplane>)
%
% shows the feature means and covariances in one plot;
% covariances are shown as ellipsoids.
%
% IN: fv   - a feature struct. fv.x should have only two feature 
%            dimensions!
%     h    - pointer to objects of the current figure
%     hyperplane - a struct containing fields w and b.
% OUT:h    - pointer to objects of the generated figure.
%
%

%kraulem 08/07

if length(varargin)>0 & isnumeric(varargin{1}),
  opt.h= varargin{1};
  if length(varargin)>1,
    opt.hyperplane= varargin{2};
  end
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'h', [], ...
                  'hyperplane', [], ...
                  'plot_samples', 1, ...
                  'plot_mean', 1, ...
                  'mean_spec', {'Marker','x', 'MarkerSize',12, 'LineWidth',2}, ...
                  'colors', {'r','g','m','b'}, ...
                  'marker_style', '.', ...
                  'common_cov', 0, ...
                  'cov_factor', 1, ...
                  'cla', 1);

if ~iscell(opt.marker_style),
  opt.marker_style= {opt.marker_style};
end

if ~isempty(opt.h),
  h = opt.h;
  if ~isstruct(h)
    h.fig = h;
  end
else 
  h.fig = gcf;
end
if strcmp(get(h.fig,'Type'),'figure')
  figure(h.fig);
end

% get mean and covariances:
x = squeeze(fv.x);
for ii = 1:size(fv.y,1)
  incl{ii} = find(fv.y(ii,:)~=0);
  cova{ii} = cov(x(1,incl{ii}),x(2,incl{ii}));
  me{ii} = [mean(x(1,incl{ii}));mean(x(2,incl{ii}))];
end
if opt.common_cov,
  cova= repmat({mean(cat(3, cova{:}), 3)}, [1 size(fv.y,1)]);
end

if opt.cla,
  cla;
end
hold on;
for ii = 1:size(fv.y,1)
  h.ellip(ii) = drawEllipse(me{ii}, opt.cov_factor*cova{ii}, opt.colors{ii});
end

if opt.plot_samples,
  for ii = 1:size(fv.y,1)
    incl{ii} = find(fv.y(ii,:)~=0);
    h.samples(ii)= plot(x(1,incl{ii}), x(2,incl{ii}), ...
                        opt.marker_style{1+mod(ii-1,length(opt.marker_style))});
    set(h.samples(ii), 'Color', opt.colors{ii});
  end
end

if opt.plot_mean,
  for ii = 1:size(fv.y,1)
    h.mean(ii)= plot(me{ii}(1), me{ii}(2), opt.mean_spec{:}, ...
                     'Color',opt.colors{ii});
  end
end

if ~isfield(h,'xl')
  h.xl = get(gca,'XLim');
  h.yl = get(gca,'YLim');
end

if ~isempty(opt.hyperplane),
  % also draw the hyperplane
  % parametric form of hyperplane: hy*w+r*u (r variable)
  u = [0 -1;1 0]*opt.hyperplane.w;
  u = u./norm(u);
  hy = -opt.hyperplane.b*opt.hyperplane.w/norm(opt.hyperplane.w)^2;
  hplen= max(diff(h.xl), diff(h.yl));
  hy1 = hy-hplen*u;
  hy2 = hy+hplen*u;
  h.hyp = line([hy1(1) hy2(1)],[hy1(2) hy2(2)]);
  set(h.hyp,'Color',[0 0 0]);
  set(gca,'XLim',h.xl,'YLim',h.yl);
end
hold off;
h.xlabel = xlabel('First Feature dimension');
h.ylabel = ylabel('Second Feature dimension');
return
