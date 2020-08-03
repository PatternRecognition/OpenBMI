function h= showTimeCoursePlusFeatureDistributions(fv, ival, hyp, varargin)
%h= showTimeCoursePlusFeatureDistributions(fv, ival, hyp, <opts>)
%
% Makes a timecourse plot in the upper panel with given interval marked,
% and draws below feature distributions for all marked intervals.
% For each class topographies are
% plotted in one row and shared the same color map scaling. 
%
% IN: fv   - continuous feature vector struct with fields
%      .x      - [2 x nSamples]-sized array of features, which will be taken for 
%                calculating the feature distributions.
%      .disp   - features which should be displayed in the timecourse.
%      .dispName - cell array with names of the displayed timecourses.
%     fv.x may also be given in form of a struct array of length length(fv.disp).
%     In this case, a cell array field fv.x.y is also required.
%     ival - [nIvals x 2]-sized array of interval, which are marked in the
%            ERP plot and for which scalp topographies are drawn.
%            When all interval are consequtive, ival can also be a
%            vector of interval borders.
%     hyp  - hyperplane to plot into the feature distributions.
%     opts - struct or property/value list of optional fields/properties:
%      .ival_color - [nColors x 3]-sized array of rgb-coded colors
%                    with are used to mark intervals and corresponding 
%                    scalps. Colors are cycled, i.e., there need not be
%                    as many colors as interval. Two are enough,
%                    default [0.75 1 1; 1 0.75 1].
%      .legend_pos - specifies the position of the legend in the ERP plot,
%                    default 0 (see help of legend for choices).
%      the opts struct is passed to plotMeanScalpPattern
%
% OUT h - struct of handles to the created graphic objects.

%% kraulem 09/05 - some code stolen from showERPplusScalpPatterns.


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lineWidth',3, ...
                  'ival_color',[0.75 1 1; 1 0.75 1], ...
                  'legend_pos', 0,...
		  'reproject',false,...
		  'lineColor',[0 1 0],...
		  'xl',[],...
		  'yl',[]);

% if isfield(opt, 'colorOrder') & isequal(opt.colorOrder,'rainbow'),
%   nChans= size(erp.y,1);
%   opt.colorOrder= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
% end
% if size(erp.x,3)>1,
%   erp= proc_average(erp);
% end

[axesStyle, lineStyle]= opt_extractPlotStyles(opt);
colMap = linspace(0,0.8,size(fv.disp,1))'*[1 1 1];
featcolMap = 	  linspace(0,0.8,2)'*[1 1 1];

if size(ival,1)==1,
  ival= [ival(1:end-1)', ival(2:end)']';
end
nIvals= size(ival,2);
%nClasses= length(fv.className);

clf;
h.ax_erp= subplotxl(2, 1, 1, 0.05, [0.07 0 0.05]);
if ~isempty(axesStyle),
  set(h.ax_erp, axesStyle{:});
end
hold on;   %% otherwise axis properties like colorOrder are lost
for ii = 1:size(fv.disp,1)
  h.plot_erp(ii) = plot(find(fv.y(ii,:)),fv.disp(ii,find(fv.y(ii,:))));
end
if length(lineStyle)>0,
  ii = 1;
  for hpp= h.plot_erp,
    set(hpp, lineStyle{:});
    set(hpp, 'Color',colMap(ii,:));
    ii = ii+1;
  end
end
set(h.ax_erp, 'box','on');

nColors= size(opt.ival_color,1);
% mark the intervals with the specified colors:
ylim = get(h.ax_erp,'yLim');
for cc = 1:nColors
  h.pat(cc) = patch(ival([1 2 2 1], cc:nColors:end),repmat(ylim([1 1 2 2])',1,length(cc:nColors:size(ival,2))),opt.ival_color(cc,:));
end
moveObjectBack(h.pat);
if ~isequal(opt.legend_pos, 'none'),
  h.leg= legend(fv.dispName, opt.legend_pos);
end

% Now plot the feature distributions of the given intervals:
% fv.x should have the correct number of dimensions:
if ~isstruct(fv.x)& ndims(fv.x)>2
  fv.x = squeeze(fv.x);
end
% normalize the normal vector of the hyperplane
norm1 = norm(hyp.w);
hyp.w = hyp.w/norm1;
hyp.b = hyp.b/norm1;
  
if ~isstruct(fv.x)
  if (size(fv.x,1) > 2 )| opt.reproject
    % project onto the norm of the hyperplane and the largest PCA component:
    % first find the dimensions that we wish to project on;
    %... and the largest PCA component of the whole dataset.
    for ii = 1:length(fv.className)
      fv1 = proc_selectClasses(fv,fv.className{1});
      covi(ii,:,:) = cov(fv1.x');
      me(:,ii) = mean(fv1.x,2);
    end
    all_cov = squeeze(covi(1,:,:)+covi(2,:,:))/2;
    [V,D] = eig(all_cov);
    N = ortho_basis([hyp.w, V(:,1)]);
    % do the projections for all participating features:
    fv1 = fv;
    fv1.x = reshape(fv1.x,[1 size(fv1.x,1) size(fv1.x,2)]);
    fv1 = proc_linearDerivation(fv1,N(:,1:2));
    fv = fv1;
    fv.x = squeeze(fv.x);
    %...and the hyperplane:
    hyp.w = N(:,1:2)'*hyp.w; 
  end  
end
for ii= 1:nIvals,
  h.ax_topo(ii)= subplotxl(2, nIvals, ii+nIvals, ...
			   [0.05 0.05 0.1], [0.07 0.03 0.05]);
  if isstruct(fv.x)
    fv1 = fv;
    ind = ival(1,ii):ival(2,ii);
    fv1.x = cat(2,fv.x(ind).x);
    fv1.y = cat(2,fv.x(ind).y);
  else
    fv1 = proc_selectEpochs(fv,ival(1,ii):ival(2,ii));
  end
  h.feat(ii) = plot_featureDistributions(fv1,h.ax_topo(ii),hyp);
  set(h.feat(ii).ellip(1),'Color',featcolMap(1,:));
  set(h.feat(ii).ellip(2),'Color',featcolMap(2,:));
end
%run it again to make the axes equal:
dum = [max(cat(1,h.feat.xl),[],1),max(cat(1,h.feat.yl),[],1),...
       min(cat(1,h.feat.xl),[],1),min(cat(1,h.feat.yl),[],1)];
xl = dum([5 2]);
yl = dum([7 4]);
if ~isempty(opt.xl)
  xl = opt.xl;
end
if ~isempty(opt.yl)
  yl = opt.yl;
end
for ii= 1:nIvals,
  set(h.ax_topo(ii),'xLim',xl,'yLim',yl);
  h.feat(ii).xl = xl;
  h.feat(ii).yl = yl;
  h = borderplot(h,ii,[5 5],opt.ival_color(rem(ii-1,size(opt.ival_color,1))+1,:));
  delete([h.feat(ii).xlabel,h.feat(ii).ylabel]);
end


pos= get(h.ax_topo(1), 'position');
yy= pos(2)+0.5*pos(4);
h.background= getBackgroundAxis;
h.text= text(0.01, yy, 'Feature Distributions');
set(h.text, 'verticalAli','top', 'horizontalAli','center', ...
	    'rotation',90, 'visible','on', ...
	    'fontSize',12, 'fontWeight','bold');

if nargout<1,
  clear h
end
return

function h = borderplot(h,index,width,color)
% plots a border around the current axis h, in a specified color.
xlim = get(h.ax_topo(index),'XLim');
ylim = get(h.ax_topo(index),'YLim');
width = [abs(diff(xlim)) abs(diff(ylim))]/100.*width;
axes(h.ax_topo(index));
h.pat(index) = patch([xlim(1),xlim(1),xlim(2),xlim(2);...
		    xlim(1),xlim(1),xlim(2),xlim(2);...
		    xlim(1),xlim(1),xlim(1)+width(1),xlim(1)+width(1);...
		    xlim(2),xlim(2),xlim(2)-width(1),xlim(2)-width(1)]',...
		     [ylim(1),ylim(1)+width(2),ylim(1)+width(2),ylim(1);...
		    ylim(2),ylim(2)-width(2),ylim(2)-width(2),ylim(2);...
		    ylim(1),ylim(2),ylim(2),ylim(1);...
		    ylim(1),ylim(2),ylim(2),ylim(1)]',...
		     color);
set(h.pat(index),'EdgeColor',color);
moveObjectBack(h.pat(index));
return