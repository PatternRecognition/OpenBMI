function H= grid_addBars(fv, varargin)
%GRID_ADDBARS - Add a Colorbar to the Bottom of Each Subplot of a Grid Plot
%
%Synopsis:
%  H= grid_addBars(FV, <opt>)
%
%Arguments:
%  FV: data structure (like epo)
%  OPT: struct or property/value list of optional properties:
%   .height: height of the colorbar
%   .h_scale: handle to the scale of the grid plot (returned by grid_plot as .scale)
%             A colorbar will be placed next to the scale.
%   ...  (TODO: describe other options)
%
%Caveat:
%  You have to make sure that the FV data fits to the data that was displayed
%  with grid_plot before (e.g. for the length of the vector in the time
%  dimension). The channels are matched automatically. FV may have less
%  channels than displayed by grid_plot.
%
%Example:
%  
% H= grid_plot(epo, mnt, defopt_erps)
% epo_rsq= proc_r_square_signed(epo);
% grid_addBars(epo_rsq, 'h_scale',H.scale)

fig_visible = strcmp(get(gcf,'Visible'),'on'); % If figure is already invisible jvm_* functions should not be called
if fig_visible
  jvm= jvm_hideFig;
end

if ndims(fv.x)>2,
  error('only one class allowed');
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'vpos', 0, ...
                 'height', 1/15, ...
                 'shiftAxes', 'ylim', ...
                 'shiftAlso', {'scale'}, ...
                 'cLim', 'auto', ...
                 'colormap', flipud(gray(32)), ...
                 'useLocalColormap', 1, ...
                 'rectify', 0, ...
                 'moveBack', 0, ...
                 'box', 'on', ...
                 'visible', 'off', ...
                 'alpha_steps_mode', 0, ...
                 'h_scale', [], ...
                 'scale_height', 0.66, ...
                 'scale_width', 0.075, ...
                 'scale_vpos', 0.25, ...
                 'scale_leftshift', 0.05, ...
                 'scale_fontSize', get(gca,'fontSize'), ...
                 'scale_digits', 4, ...
                 'scale_unit', '', ...
                 'chans', 'plus');

if min(fv.x(:))<0,
  if isdefault.cLim,
    opt.cLim= 'sym';
  end
end
if isdefault.colormap && isequal(opt.cLim,'sym'),
  opt.colormap= cmap_posneg(21);
end

if isdefault.alpha_steps_mode && ...
      isfield(fv, 'className') && ...
      ~isempty(findstr(fv.className{1},'alpha-steps')),
  opt.alpha_steps_mode= 1;
end
if opt.alpha_steps_mode,
  nValues= length(fv.alpha);
  [opt, isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'cLim', [0 nValues], ...
                            'colormap', flipud(gray(nValues+1)), ...
                            'scale_unit', '%');
end
if isdefault.scale_unit && isfield(fv, 'yUnit'),
  opt.scale_unit= fv.yUnit;
end
if isdefault.scale_vpos && isempty(opt.scale_unit),
  opt.scale_vpos= 0.5;
end
if isdefault.visible && strcmpi(opt.box, 'on'),
  opt.visible= 'on';
end
if isdefault.useLocalColormap,
  if iscolormapused && ~isequal(opt.colormap, get(gcf, 'colormap')),
    opt.useLocalColormap= 1;
  end
end

[axesStyle, lineStyle]= opt_extractPlotStyles(opt);

if opt.useLocalColormap,
  iswhite= find(all(opt.colormap==1,2));
  if ~isempty(iswhite),
    opt.colormap(iswhite,:)= 0.9999*[1 1 1];
  end
  H.image= 'sorry: fakeimage - no handle';
%  if ~isempty(opt.h_scale),
%    warning('sorry, no colorbar in local colormap mode (so far)');
%    opt.h_scale= [];
%  end
else
  set(gcf, 'colormap',opt.colormap);
end

if opt.rectify,
  fv= proc_rectifyChannels(fv);
end

if isnumeric(opt.chans)
  ax= opt.chans;
else
  ax= grid_getSubplots(opt.chans);
end

%% For image_local_cmap we have to determine the cLim in advance.
%% Therefore we need to determine the depicted channels.
%% (If only image was used, this functions would be simpler.)
clab= cell(1, length(ax));
for ii= 1:length(ax),
  clab{ii}= getfield(get(ax(ii), 'userData'), 'chan');
  if iscell(clab{ii}),
    clab{ii}= clab{ii}{1};  %% for multiple channels per ax, choose only
                            %% the first one
  end
end
if strcmpi(opt.cLim, 'auto'),
  ci= chanind(fv, clab);
  mi= min(min(fv.x(:,ci)));
  if mi>=0 && isdefault.cLim,
    warning('know-it-all: switching to cLim mode ''0tomax''');
    opt.cLim= '0tomax';
  else
    opt.cLim= [mi max(max(fv.x(:,ci)))];
  end
elseif strcmpi(opt.cLim, 'sym'),
  ci= chanind(fv, clab);
  mi= min(min(fv.x(:,ci)));
  ma= max(max(fv.x(:,ci)));
  mm= max(abs(mi), ma);
  opt.cLim= [-mm mm];
end
if strcmpi(opt.cLim, '0tomax'),
  ci= chanind(fv, clab);
  opt.cLim= [0 max(max(fv.x(:,ci)))];
end

jj= 0;
for ii= 1:length(ax),
  set(ax(ii), 'YLimMode','manual');
  ud= get(ax(ii), 'userData');
  if iscell(ud.chan),
    ud.chan= ud.chan{1};  %% for multiple channels per axis take only the first
  end
  ci= chanind(fv, ud.chan);
  if isempty(ci) && isempty(strmatch(ud.chan,opt.shiftAlso,'exact')),
    continue;
  end
  pos= get(ax(ii), 'position');
  bar_pos= [pos(1:3) opt.height*pos(4)];
  bar_pos(2)= pos(2) + opt.vpos*(1-opt.height)*pos(4);
  switch(opt.shiftAxes)
    case {1,'position'},
     if opt.vpos<0.5,
       new_pos= [pos(1) pos(2)+bar_pos(4) pos(3) pos(4)*(1-opt.height)];
     else
       new_pos= [pos(1:3) pos(4)*(1-opt.height)];
       if opt.vpos==1,
         axis_raiseTitle(ax(ii), opt.height);
       end
     end
     set(ax(ii), 'position',new_pos);
    case {2,'ylim'},
     yLim= get(ax(ii), 'yLim');
     if opt.vpos<0.5,
       yLim(1)= yLim(1) - opt.height*diff(yLim);
     else
       yLim(2)= yLim(2) + opt.height*diff(yLim);
     end
     set(ax(ii), 'yLim',yLim);
   otherwise,
    error('shiftAxes policy not known');
  end
  if isempty(ci),
    continue;
  end
  jj= jj+1;
  H.ax(jj)= axes('position', bar_pos);
  set(H.ax(jj), axesStyle{:});
  hold on;      %% otherwise axis properties like colorOrder are lost
  if opt.useLocalColormap,
    %%StHauf
    %%an error occured in image_local_map, 
    %%when signed-values were to be plotted
    %%and some values exceeded clim
    fv.x(fv.x > opt.cLim(2)) = opt.cLim(2);
    fv.x(fv.x < opt.cLim(1)) = opt.cLim(1);
    %%
    image_local_cmap(fv.x(:,ci)', opt.colormap, 'cLim',opt.cLim);
  else
%    H.image(jj)= image(fv.t, 1, fv.x(:,ci)', 'cDataMapping','scaled');
    H.image(jj)= image(fv.x(:,ci)', 'cDataMapping','scaled');
  end
  set(H.ax(jj), axesStyle{:});
  ud= struct('type','ERP plus: bar', 'chan',vec2str(fv.clab(ci)));
  set(H.ax(jj), 'userData',ud);
  hold off;
  if strcmp(get(H.ax(jj), 'box'), 'on'),
    h= axis_redrawFrame(H.ax(jj), 'LineWidth',0.3);
    if jj==1,
      H.frame= h;
    else
      H.frame(:,jj)= h;
    end
  end
end
if diff(opt.cLim)==0, opt.cLim(2)= opt.cLim(2)+eps; end
%set(H.ax, 'xLim',fv.t([1 end]), 'xTick',[], 'yTick',[], 'cLim',opt.cLim);
set(H.ax, 'xLim',[0.5 size(fv.x,1)+0.5], 'xTick',[], 'yTick',[], ...
          'cLim',opt.cLim);

if opt.moveBack,
  moveObjectBack(H.ax);
end

if ~isempty(opt.h_scale),
  pos= get(opt.h_scale.ax, 'position');
  dh= pos(4)*(1-opt.scale_height)*opt.scale_vpos;
  pos_cb= [pos(1)+(1-opt.scale_width-opt.scale_leftshift)*pos(3) pos(2)+dh ...
           opt.scale_width*pos(3) opt.scale_height*pos(4)];
  H.scale.ax= axes('position', pos_cb);
  cLim= opt.cLim;
  colbar= linspace(cLim(1), cLim(2), size(opt.colormap,1));
  if opt.useLocalColormap,
    image_local_cmap(colbar', opt.colormap, 'cLim',opt.cLim);
  else
    H.scale.im= imagesc(1, colbar, [1:size(opt.colormap,1)]');
  end
  axis_redrawFrame(H.scale.ax);
  if opt.alpha_steps_mode,
    yTick= 0:nValues;
    if opt.useLocalColormap, yTick= yTick+1; end %% Just a hack...
    set(H.scale.ax, 'yTick',yTick, 'yTickLabel',100*[1 fv.alpha]);
  else
    ticks= goodContourValues(cLim(1), cLim(2), -3);
    tickLabels= trunc(ticks, opt.scale_digits);
    if opt.useLocalColormap,
      YLim= get(gca,'YLim');
      ticks= (ticks-cLim(1))*diff(YLim)/diff(cLim)+YLim(1);
    end
    set(H.scale.ax, 'yTick',ticks, 'yTickLabel',tickLabels);
  end
  set(H.scale.ax, 'xTick',[], 'tickLength',[0 0], 'yDir','normal', ...
                  'fontSize',opt.scale_fontSize);
  if ~isempty(opt.scale_unit),
    yLim= get(H.scale.ax, 'yLim');
    H.scale.unit= text(1, yLim(2), opt.scale_unit);
    set(H.scale.unit, 'horizontalAli','center', 'verticalAli','bottom');
  end
end


if nargout==0,
  clear H;
end

if fig_visible
  jvm_restoreFig(jvm, opt);
end
