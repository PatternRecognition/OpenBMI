function [H, Ctour]= scalpPlot(mnt, w, varargin);
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
%  .colAx:        'range', 'sym' (default), '0tomax', 'minto0',
%                 or [minVal maxVal]
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
%  .extrapolate  Default value (1) extends the scalp plot to the peripheral
%                areas where no
%                channels are located. Value (0) turns off extrapolation.
%  .extrapolateToMean: Default value (1) paints peripheral area
%                in color of average (zero?) value. Needs .extrapolate 
%                activated. 
%  .extrapolateToZero: Value (1) paints peripheral area in "zero"-color.
%                Needs .extrapolate activated. 
%  .renderer:    The function used for rendering the scalp map, 'pcolor'
%                or 'contourf' (default).
%  .contourfLevels: number of levels for contourf function (default 100).
%  .offset       a vector of length 2  -  [x_offset y_offset]
%                normally, the scalpplot is drawn centered at the origin,
%                i.e. [x_offset y_offset] = [0 0] by default
%
%Output:
% H:     handle to several graphical objects
% Ctour: Struct of contour information
%
%See also scalpPatterns, scalpEvolution.

% Author(s): Benjamin Blankertz, Aug 2000; Feb 2005, Matthias
% Added contourf: Matthias Treder 2010
% Added "extrapolateToZero" option: Simon Scholler, 2011
% "offset" option added: Simon Scholler, 2011

fig_visible = strcmp(get(gcf,'Visible'),'on'); % If figure is already invisible jvm_* functions should not be called
if fig_visible
  jvm= jvm_hideFig;
end

opt= propertylist2struct(varargin{:});
[opt_orig, isdefault]= ...
    set_defaults(opt, ...
                 'w_clab', [], ...
                 'linespec', {'k'}, ...
                 'contour', 5, ...
                 'contour_policy', 'levels', ...
                 'contour_lineprop', {'linewidth',1}, ...
                 'contour_labels', 0, ...
                 'ticksAtContourLevels', 1, ...
                 'markcontour', [], ...
                 'markcontour_lineprop', {'linewidth',2}, ...
                 'shading', 'flat', ...
                 'resolution', 40, ...
                 'extrapolate', 1, ...
                 'colAx', 'sym', ...
                 'shrinkColorbar', 0, ...
                 'newcolormap', 0, ...
                 'interpolation', 'linear', ...
                 'scalePos', 'vert',...
                 'extrapolation', 1,...
                 'extrapolateToMean',1,...
                 'extrapolateToZero', 0, ...
                 'renderer','contourf', ...
                 'contourfLevels',50, ...
                 'contourMargin', 0, ...
                 'offset', [0 0]);
opt= setfield(opt_orig, 'fig_hidden', 1);

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
if isempty(opt.resolution)
  oldUnits= get(H.ax, 'units');
  set(H.ax, 'units', 'normalized');
  pos= get(H.ax, 'position');
  set(H.ax, 'units', oldUnits);
  opt.resolution= max(20, 60*pos(3)); 
end

% Allow radius of scalp data to go beyond scalp outline (>1)
maxrad = max(1,max(max(abs(mnt.x)),max(abs(mnt.y)))) + opt.contourMargin; 

%% Extrapolation
if opt.extrapolate,
  xx= linspace(-maxrad, maxrad, opt.resolution);
  yy= linspace(-maxrad, maxrad, opt.resolution)';
  if opt.extrapolateToMean
    xe_add = cos(linspace(0,2*pi,opt.resolution))'*maxrad;
    ye_add = sin(linspace(0,2*pi,opt.resolution))'*maxrad;
    w_add = ones(length(xe_add),1)*mean(w);
    xe = [xe;xe_add];
    ye = [ye;ye_add];
    w = [w;w_add];
  end
  if opt.extrapolateToZero
    xe_add = cos(linspace(0,2*pi,opt.resolution))';
    ye_add = sin(linspace(0,2*pi,opt.resolution))';
    xe = [xe;xe_add];
    ye = [ye;ye_add];
    w = [w; zeros(length(xe_add),1)];
  end
  
else
  xx= linspace(min(xe), max(xe), opt.resolution);
  yy= linspace(min(ye), max(ye), opt.resolution)';
end

if opt.extrapolate,
  wstate= warning('off');
%  [xg,yg,zg]= griddata(xe, ye, w, xx, yy, 'invdist');
  [xg,yg,zg]= griddata(xe, ye, w, xx, yy);
  warning(wstate);
  margin = maxrad +opt.contourMargin;
  headmask= (sqrt(xg.^2+yg.^2)<=margin);
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

% contour line coordinated
xgc= xg+opt.offset(1);
ygc= yg+opt.offset(2);
zgc= zg;
if ~isempty(strmatch(lower(opt.shading), {'flat','faceted'})) ... 
    && strcmp(opt.renderer,'pcolor'),
  %% in shading FLAT last row/column is skipped, so add one
  xg= [xg-xs/2, xg(:,end)+xs/2];
  xg= [xg; xg(end,:)];
  yg= [yg, yg(:,end)]-ys/2;
  yg= [yg; yg(end,:)+ys];
  zg= [zg, zg(:,end)];
  zg= [zg; zg(end,:)];
end

%% Render using pcolor or contourf
xg= xg+opt.offset(1);
yg= yg+opt.offset(2);
if strcmp(opt.renderer,'pcolor')
  H.patch= pcolor(xg, yg, zg);
else
  [pts,H.patch]= contourf(real(xg), real(yg), real(zg), opt.contourfLevels,'LineStyle','none');
  % *** Hack to enforce cdatamappig = scaled in colorbarv6.m by introducing
  % a useless patch object
  hold on
  patch([0 0],[0 0],[1 2]);
  ccc = get(gca,'children');
  set(ccc(1),'Visible','off');
end

%%
tight_caxis= [min(zg(:)) max(zg(:))];
if isequal(opt.colAx, 'sym'),
  zgMax= max(abs(tight_caxis));
  H.cLim= [-zgMax zgMax];
elseif isequal(opt.colAx, 'range'),
  H.cLim= tight_caxis;
elseif isequal(opt.colAx, 'rangesymcol'),
  H.cLim= tight_caxis;
  nColors= size(get(gcf, 'Colormap'), 1);
  nColorsNeg= round(nColors*max(0, -H.cLim(1))/diff(H.cLim));
  nColorsPos= round(nColors*max(0, H.cLim(2))/diff(H.cLim));
  colormap(cmap_posneg_asym(nColorsNeg, nColorsPos));
elseif isequal(opt.colAx, '0tomax'),
  H.cLim= [0.0001*diff(tight_caxis) max(tight_caxis)];
elseif isequal(opt.colAx, 'minto0'),
  H.cLim= [min(tight_caxis) -0.0001*diff(tight_caxis)];
elseif isequal(opt.colAx, 'zerotomax'),
  H.cLim= [0 max(tight_caxis)];
elseif isequal(opt.colAx, 'mintozero'),
  H.cLim= [min(tight_caxis) 0];
else
  H.cLim= opt.colAx;
end
if diff(H.cLim)==0, H.cLim(2)= H.cLim(2)+eps; end
set(gca, 'CLim',H.cLim);

if strcmp(opt.renderer,'pcolor')
  shading(opt.shading);
end

hold on;
if ~isequal(opt.contour,0),
  if length(opt.contour)>1,
    ctick= opt.contour;
    v= ctick;
%    v= v(min(find(v>min(H.cLim))):max(find(v<max(H.cLim))));
  else
%    mi= min(tight_caxis);
%    ma= max(tight_caxis);
    mi= min(H.cLim);
    ma= max(H.cLim);
    switch(opt.contour_policy),
     case {'levels','strict'},
      ctick= linspace(mi, ma, abs(opt.contour)+2);
      v= ctick([2:end-1]);
     case 'withinrange',
      ctick= linspace(min(tight_caxis), max(tight_caxis), abs(opt.contour)+2);
      v= ctick([2:end-1]);
     case 'spacing',
      mm= max(abs([mi ma]));
      v= 0:opt.contour:mm;
      v= [fliplr(-v(2:end)), v];
      ctick= v(find(v>=mi & v<=ma));
      v(find(v<=mi | v>=ma))= [];
     case 'spacing_compatability',
      v= floor(mi):opt.contour:ceil(ma);
      ctick= v(find(v>=mi & v<=ma));
      v(find(v<=mi | v>=ma))= [];
     case 'choose',
      ctick= goodContourValues(mi, ma, -abs(opt.contour));
      v= ctick;
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
    [c,H.contour]= contour(real(xgc), real(ygc), real(zgc), v_tmp, 'k-');
    set(H.contour, opt.contour_lineprop{:});
    if opt.contour_labels, %% & length(H.contour)>1,
      clabel(c,H.contour);
    end
  end
  H.contour_heights= v;
else
  H.contour_heights = [];   %% recently added
end
if ~isempty(opt.markcontour),
  if length(opt.markcontour)==1,
    v_tmp= [1 1]*opt.markcontour;
  else
    v_tmp= opt.markcontour;
  end
  [c,H.markcontour]= contour(xgc, ygc, zgc, v_tmp, 'k-');
  set(H.markcontour, opt.markcontour_lineprop{:});
end

%% Scalp outline
H= drawScalpOutline(mnt, opt, 'H',H, 'dispChans',dispChans);

if strcmp(opt.scalePos, 'none'),
  H.cb= [];
else
  if verLessThan('matlab', '7.14')
    H.cb= colorbarv6(opt.scalePos);
  else
    H.cb= colorbar(opt.scalePos);
  end
  if opt.ticksAtContourLevels && opt.contour,
    if strcmpi(opt.scalePos, 'vert'),
      set(H.cb, 'yLim',H.cLim, 'yTick', ctick);
%      ylabel(opt.yUnit);
      if opt.shrinkColorbar>0,
        cbpos= get(H.cb, 'Position');
        cbpos(2)= cbpos(2) + cbpos(4)*opt.shrinkColorbar/2;
        cbpos(4)= cbpos(4) - cbpos(4)*opt.shrinkColorbar;
        set(H.cb, 'Position',cbpos);
      end
    else
      set(H.cb, 'xLim',H.cLim, 'xTick', ctick);
%      xlabel(opt.yUnit);
      if opt.shrinkColorbar>0,
        cbpos= get(H.cb, 'Position');
        cbpos(1)= cbpos(1) + cbpos(1)*opt.shrinkColorbar/2;
        cbpos(3)= cbpos(3) - cbpos(3)*opt.shrinkColorbar;
        set(H.cb, 'Position',cbpos);
      end
    end
  end
  if ~isempty(H.contour_heights) && opt.ticksAtContourLevels,
    set(H.cb, 'YTick',H.contour_heights);
  end
end
if opt.newcolormap,
  fig_acmAdaptCLim(acm);
  set(H.cb, 'yLim',H.cLim); %% otherwise ticks at the border of the
                            %% colorbar might get lost
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

if fig_visible
  jvm_restoreFig(jvm, opt_orig);
end
