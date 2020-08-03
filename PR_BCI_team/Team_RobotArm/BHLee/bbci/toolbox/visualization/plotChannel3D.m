function H= plotChannel3D(epo, clab, varargin)
%plotChannel3D - Plot the classwise averages of one channel. Plots 3D data,
%i.e. frequency x time x amplitude.
%
%Usage:
% H= plotChannel3D(EPO, CLAB, <OPT>)
%
%Input:
% EPO  - Struct of epoched signals, see makeEpochs
% CLAB - Name (or index) of the channel to be plotted.
% OPT  - struct or property/value list of optional properties:
%  .xUnit  - unit of x axis, default 'ms'
%  .yUnit  - unit of y axis, default epo.unit if this field
%                     exists, 'Hz' otherwise
%  .yDir     - 'normal' (low frequencies at bottom) or 'reverse'
%  .freq     - A vector giving lowest and highest frequency. If not specified, 
%              and epo.wave_freq exists the corresponding values are taken;
%              otherwise [1 size(epo.x,1)] is taken.
%  .plotRef  - if 1 plot reference interval (default 0)
%  .refY     - y position of reference line 
%  .refWhisker - length of whiskers (vertical lines)
%  .ref*     - with * in {'LineStyle', 'LineWidth', 'Color'}
%              selects the appearance of the reference interval.
%  .colormap - specifies the colormap for depicting amplitude. Give either
%              a string or a x by 3 color matrix (default 'jet')
%  .colLim   - Define the color (=amplitude) limits. If empty (default), 
%              limits correspond to the data limits.
%  .colLimPolicy - if 'sym', color limits are symmetric (so that 0
%              corresponds to the middle of the colormap) (default
%              'normal')
%  .xGrid, ... -  many axis properties can be used in the usual
%                 way
%  .grid_over_patches - if 1 plot grid (default 0)
%  .title   - Title of the plot to be displayed above the axis. 
%             If OPT.title equals 1, the channel label is used.
%  .title*  - with * in {'Color', 'FontWeight', 'FontSize'}
%             selects the appearance of the title.
%  .titleY  - if set, the title is displayed within the axis, with its 
%             Y position corresponding to titleY (default [])
%  .zeroLine  - draw an axis along the y-axis at x=0
%  .zeroLine*  - with * in {'Color','Style'} selects the
%                drawing style of the axes at x=0/y=0
%
%Output:
% H - Handle to several graphical objects.
%
%Do not call this function directly, rather use the superfunction
%plotChannel. This function is an adapted version of plotChannel2D.
%
%See also plotChannel2D, grid_plot.

% Author(s): Matthias Treder Aug 2010

opt= propertylist2struct(varargin{:});

if ~isfield(opt,'freq') && isfield(epo,'wave_freq')
  opt.freq = [epo.wave_freq(1) epo.wave_freq(end)];
end

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'axisType', 'box', ...
                 'yDir', 'normal', ...
                 'xGrid', 'on', ...
                 'yGrid', 'on', ...
                 'box', 'on', ...
                 'xUnit', '[ms]', ...
                 'yUnit', '[Hz]', ...
                 'freq', [1 size(epo.x,1)], ...
                 'plotRef', 0, ...
                 'refCol', 0.6, ...
                 'refLineStyle', '-', ...
                 'refLineWidth', 2, ...
                 'refY', [], ...
                 'refWhisker', [], ...
                 'zeroLine', 1, ...
                 'zeroLineColor', 0.5*[1 1 1], ...
                 'zeroLineStyle', '-', ...
                 'lineWidth', 2, ...
                 'title', 1, ...
                 'titleColor', 'k', ...
                 'titleFontSize', get(gca,'fontSize'), ...
                 'titleFontWeight', 'normal', ...
                 'titleY', [], ...
                 'smallSetup', 0, ...
                 'multichannel_title_opts', {}, ...
                 'colormap','jet',...
                 'colLim', [], ...
                 'colLimPolicy', 'normal', ...
                 'grid_over_patches', 0, ...
                 'oversizePlot',1);

if max(sum(epo.y,2))>1,
  epo= proc_average(epo);
else
  % epo contains already averages (or single trials)
  % sort classes
  [tmp,si]= sort([1:size(epo.y,1)]*epo.y);
  epo.y= epo.y(:,si);  % should be an identity matrix now
  epo.x= epo.x(:,:,:,si);
end

chan= chanind(epo, clab);
nChans= length(chan);
nClasses= size(epo.y, 1);

if nChans==0,
  error('channel not found'); 
elseif nChans>1,
  opt_plot= {'zeroLine',0, 'title',0, ...
            'grid_over_patches',0};
  tit= cell(1, nChans);
  for ic= 1:nChans,
    if ic==nChans,
      opt_plot([4 6])= {1};
    end
    ils= mod(ic-1, length(opt.channelLineStyleOrder))+1;
    H{ic}= plotChannel2D(epo, chan(ic), opt_rmifdefault(opt, isdefault), ...
                       opt_plot{:}, ...
                       'lineStyle',opt.channelLineStyleOrder{ils});
    hold on;
    tit{ic}= sprintf('%s (%s)', epo.clab{chan(ic)}, ...
                     opt.channelLineStyleOrder{ils});
    opt_plot{2}= 0;
  end
  hold off;
  H{1}.leg= NaN;
  H{1}.title= axis_title(tit, opt.multichannel_title_opts{:});
  ud= struct('type','ERP', 'chan',{epo.clab(chan)}, 'hleg',H{1}.leg);
  set(gca, 'userData', ud);
  return;
end

%% Post-process opt properties
if isequal(opt.title, 1),
  opt.title= epo.clab(chan);
end

if isdefault.xUnit && isfield(epo, 'xUnit'),
  opt.xUnit= ['[' epo.xUnit ']'];
end
if isdefault.yUnit && isfield(epo, 'yUnit'),
  opt.yUnit= ['[' epo.yUnit ']'];
end

if isdefault.refY,
  opt.refY = opt.freq(1) + .9 * diff(opt.freq);
end

if isdefault.refWhisker,
  opt.refWhisker = .05 * diff(opt.freq);
end

if length(opt.refCol)==1,
  opt.refCol= opt.refCol*[1 1 1];
end

%% Set missing optional fields of epo to default values
if ~isfield(epo, 't'),
  epo.t= 1:size(epo.x,2);
end

%% Plot data, zero line, ref ival, grid
H.ax= gca; cla

if ~isempty(opt.colLim)
  if strcmp(opt.colLimPolicy,'sym')  % make color limits symmetric
    cm = abs(max(opt.colLim));
    opt.colLim = [-cm cm];
  end
  H.plot= imagesc([epo.t(1) epo.t(end)],opt.freq, squeeze(epo.x(:,:,chan,:)), ...
    opt.colLim);
else
  H.plot= imagesc([epo.t(1) epo.t(end)],opt.freq, squeeze(epo.x(:,:,chan,:)));

end
hold on;      

if opt.zeroLine,
  line([0 0], get(gca,'YLim'), ...
                'Color',opt.zeroLineColor, 'lineStyle',opt.zeroLineStyle);
end

% Plot ref ival
if opt.plotRef && isfield(epo, 'refIval'),
  xx = epo.refIval;
  yy = opt.refY * [1 1];
  lopt = {'LineStyle',opt.refLineStyle, 'LineWidth',opt.refLineWidth, ...
    'Color', opt.refCol};
  line(xx,yy,lopt{:});
  line([xx(1) xx(1)],[yy(1)-opt.refWhisker yy(1)+opt.refWhisker], ...
    lopt{:});
  line([xx(end) xx(end)],[yy(1)-opt.refWhisker yy(1)+opt.refWhisker], ...
    lopt{:});
end

if opt.grid_over_patches,
  grid_over_patches(copy_struct(opt, 'xGrid','yGrid'));
end

%% More layout settings
colormap(opt.colormap);
set(gca,'YDir',opt.yDir)
H.leg = NaN;

%% Title and labels
if ~isequal(opt.title, 0),
  if isempty(opt.titleY)
    H.title= title(opt.title);
    set(H.title, 'color',opt.titleColor, ...
                 'fontWeight',opt.titleFontWeight, ...
                 'fontSize',opt.titleFontSize);
  else
    xx = sum(get(gca,'XLim'))/2; % middle
    H.title = text(xx,opt.titleY, opt.title, ...
                 'color',opt.titleColor, ...
                 'fontWeight',opt.titleFontWeight, ...
                 'fontSize',opt.titleFontSize, ...
                 'HorizontalAlignment','center');
    
  end
end

H.xlabel = xlabel(opt.xUnit);
H.ylabel = ylabel(opt.yUnit);

% if ~isempty(H.hidden_objects),
%   moveObjectBack(H.hidden_objects);
% % If we hide handles, those objects may pop to the front again,
% % e.g., when another object is moved to the background with moveObjetBack
% %  set(H.hidden_objects, 'handleVisibility','off');
% end
ud= struct('type','ERP', 'chan',epo.clab{chan}, 'hleg',H.leg);
set(H.ax, 'userData', ud);

if nargout==0,
  clear H,
end
