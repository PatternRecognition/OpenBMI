function [H, Ctour]= scalpPlotPlus(mnt, w, varargin);

error('in construction')

%SCALPPLOT - Display a vector as scalp topography
%
%Description:
% This is the low level function for displaying a scalp topography.
% In many cases it is more wise to use one of the other scalp*
% functions.
%
%Usage:
% H= scalpPlot(MNT, W, <OPT>)
%
%Input:
% MNT: An electrode montage, see getElectrodePositions
% W:   Vector to be displayed as scalp topography. The length of W must
%      concide with the length of MNT.clab or the number of non-NaN
%      entries of MNT.x, or OPT must include a field 'w_clab'.
%      Warning: when you do not specify OPT.w_clab you have to make sure
%      that the entries of W are matching with MNT.clab, or
%      MNT.clab(find(~isnan(MNT.x))).
% OPT: struct or property/value list of optional properties:
%  .colAx:        'range', 'sym' (default), '0tomax', or [minVal maxVal]
%  .scalePos:     Placement of the colorbar 'horiz', 'vert' (deafult), 
%                 or 'none'
%  .crossSize:    Size of the crosses marking electrode positions when
%                 no channel names are displayed; default 2.
%  .contour:      Specifies at what heights contour lines are drawn.
%                 If 'contour' is a vector, its entries define the
%                 heights. If is a scalar it specifies
%                 - according to 'contour_policy' - the number
%                 of or the spacing between contour levels. To display
%                 no contour lines set 'contour' to 0 (not []!).
%  .contour_policy: 'levels' (default): 'contour' specifies exactly the
%                 number of contour levels to be drawn, or
%                 'spacing': 'contour' specifies the spacing between two
%                 adjacent height levels, or
%                 'choose': '.contour' specifies approximately the
%                 number of height levels to be drawn, but the function
%                 'goodContourValues' is called to find nice values.
%  .resolution:   default 40. Number of steps around circle used for
%                 plotting the scalp.
%  .showLabels:   Display channel names (1) or not (0), default 0.
%  .fontSize:     Used for displaying channel names
%  .minorFontSize: Used for displaying channel names that are longer than
%                 3 characters.
%  .markerSize:   Of the circle around the channel name
%  .mark_channels: Specify channels to be ephasized, e.g., {'C3','C4'}.
%  .mark_properties: Marker properties used (1) for the crosses of the
%                channels that are to be marked if showLabel=0, or 
%                (2) for the channel names if showLabel=1.
%  .shading:     Shading method for the pcolor plot, default 'flat'.
%                Use 'interp' to get nice, smooth plots. But saving
%                needs more space.
%  .linespec:    Used to draw the outline of the scalp, default {'k'}.
%  .extrapolate  Default value (1) extends the scalp plot to the peripheral areas where no
%                channels are located. Value (0) turns off extrapolation.
%  .extrapolateToMean: Default value (1) paints peripheral area
%                in color of average (zero?) value. Added by
%                Matthias. Needs .extrapolate activated. Value (0) keeps
%                outermost values for extrapolation.
%
%Output:
% H:     handle to several graphical objects
% Ctour: Struct of contour information
%
%See also scalpPatterns, scalpEvolution.

% Author(s): Benjamin Blankertz, Aug 2000; Feb 2005, Matthias

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'w_clab', [], ...
                 'linespec', {'k'}, ...
                 'contour', 5, ...
                 'contour_policy', 'levels', ...
                 'contour_lineprop', {'linewidth',1}, ...
                 'contour_labels', 0, ...
                 'ticksAtContourLevels', 1, ...
                 'shading', 'flat', ...
                 'resolution', 40, ...
                 'extrapolate', 0, ...
                 'colAx', 'sym', ...
                 'newcolormap', 0, ...
                 'interpolation', 'linear', ...
                 'scalePos', 'vert',...
		 'extrapolation', 1,...
		 'extrapolateToMean',1);

if opt.newcolormap,
  acm= fig_addColormap(opt.colormap);
elseif isfield(opt, 'colormap'),
  colormap(opt.colormap);
end

if opt.extrapolate,
  if isdefault.linespec,
    opt.linespec= {'Color','k', 'LineWidth',3};
  end
  if isdefault.resolution,
    opt.resolution= 101;
  end
end

w= w(:);
if ~isempty(opt.w_clab),
  mnt= mnt_adaptMontage(mnt, opt.w_clab);
  if length(mnt.clab)<length(opt.w_clab),
    error('some channels of opt.w_clab not found in montage');
  end
end
if length(w)==length(mnt.clab),
  dispChans= find(~isnan(mnt.x));
  w= w(dispChans);
else
  dispChans= find(~isnan(mnt.x));
  if length(w)~=length(dispChans),
    error(['length of w must match # of displayable channels, ' ...
           'i.e. ~isnan(mnt.x), in mnt']);
  end
end
xe= mnt.x(dispChans);
ye= mnt.y(dispChans);

H.ax= gca;
oldUnits= get(H.ax, 'units');
set(H.ax, 'units', 'normalized');
pos= get(H.ax, 'position');
set(H.ax, 'units', oldUnits);
if isempty(opt.resolution), opt.resolution= max(20, 60*pos(3)); end
if opt.extrapolate,
  xx= linspace(-1, 1, opt.resolution);
  yy= linspace(-1, 1, opt.resolution)';
  if opt.extrapolateToMean
    xe_add = cos(linspace(0,2*pi,opt.resolution))';
    ye_add = sin(linspace(0,2*pi,opt.resolution))';
    w_add = ones(length(xe_add),1)*mean(w);
    xe = [xe;xe_add];
    ye = [ye;ye_add];
    w = [w;w_add];
  end
else
  xx= linspace(min(xe), max(xe), opt.resolution);
  yy= linspace(min(ye), max(ye), opt.resolution)';
end

if opt.extrapolate,
  [xg,yg,zg]= griddata(xe, ye, w, xx, yy, 'invdist');
  headmask= (sqrt(xg.^2+yg.^2)<=1);
  imaskout= find(~headmask);
  zg(imaskout)= NaN;
else
  if strcmp(opt.interpolation, 'invdist'),
    %% get the convex hull from linear interpolation
    [dmy,dmy,zconv]= griddata(xe, ye, w, xx, yy, 'linear');
    imaskout= find(isnan(zconv(:)));
    [xg,yg,zg]= griddata(xe, ye, w, xx, yy, opt.interpolation);
    zg(imaskout)= NaN;
  else
    [xg,yg,zg]= griddata(xe, ye, w, xx, yy, opt.interpolation);
  end
end
xs= xg(1,2)-xg(1,1);
ys= yg(2,1)-yg(1,1);

xgc= xg;
ygc= yg;
zgc= zg;
if ~isempty(strmatch(lower(opt.shading), {'flat','faceted'})),
  %% in shading FLAT last row/column is skipped, so add one
  xg= [xg-xs/2, xg(:,end)+xs/2];
  xg= [xg; xg(end,:)];
  yg= [yg, yg(:,end)]-ys/2;
  yg= [yg; yg(end,:)+ys];
  zg= [zg, zg(:,end)];
  zg= [zg; zg(end,:)];
end

H.patch= pcolor(xg, yg, zg);
tight_caxis= [min(zg(:)) max(zg(:))];
if isequal(opt.colAx, 'sym'),
  zgMax= max(abs(tight_caxis));
  H.cLim= [-zgMax zgMax];
elseif isequal(opt.colAx, 'range'),
  H.cLim= tight_caxis;
elseif isequal(opt.colAx, '0tomax'),
  H.cLim= [0.0001*diff(tight_caxis) max(tight_caxis)];
else
  H.cLim= opt.colAx;
end
set(gca, 'CLim',H.cLim);
shading(opt.shading);

hold on;
if ~isequal(opt.contour,0),
  if length(opt.contour)>1,
    v= opt.contour;
%    v= v(min(find(v>min(H.cLim))):max(find(v<max(H.cLim))));
  else
%    mi= min(tight_caxis);
%    ma= max(tight_caxis);
    mi= min(H.cLim);
    ma= max(H.cLim);
    switch(opt.contour_policy),
     case {'levels','strict'},
      v= linspace(mi, ma, abs(opt.contour)+2);
      v([1 end])= [];
     case 'withinrange',
      v= linspace(min(tight_caxis), max(tight_caxis), abs(opt.contour)+2);
      v([1 end])= [];
     case 'spacing',
      v= floor(mi):opt.contour:ceil(ma);
      v(find(v<=mi | v>=ma))= [];
     case 'choose',
      v= goodContourValues(mi, ma, -abs(opt.contour));
     otherwise
      error('contour_policy not known');
    end
  end
  v_tmp= v;
  if length(v)==1,
    v_tmp= [v v];
  end
  if isempty(v),
    H.contour= [];
  else
    [c,H.contour]= contour(xgc, ygc, zgc, v_tmp, 'k-');
    set(H.contour, opt.contour_lineprop{:});
    if opt.contour_labels, %% & length(H.contour)>1,
      clabel(c,H.contour);
    end
  end
end

H= drawScalpOutline(mnt, opt, 'H',H, 'dispChans',dispChans);

if strcmp(opt.scalePos, 'none'),
  H.cb= [];
else,
  H.cb= colorbarv6(opt.scalePos);
  if length(opt.contour)>1 & opt.ticksAtContourLevels,
    set(H.cb, 'yTick',unique(opt.contour));
  end
end
if opt.newcolormap,
  fig_acmAdaptCLim(acm);
end
axis('off');

if nargout==0,
  clear H;
end
if nargout>=2,
  if ~exist('c', 'var'),
    c= [];
  end
  Ctour= struct('xgrid',xg, 'ygrid',yg, 'zgrid',zg, ...
                'values',v, 'matrix',c);
end
