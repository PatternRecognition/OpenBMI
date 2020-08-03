function H= grid_plot_specgram(epo, mnt, varargin)
%GRID_PLOT - Classwise averaged epochs in a grid layout
%
%Synopsis:
% H= grid_plot(EPO, MNT, <OPT>)
%
%Input:
% EPO: struct of epoched signals, see makeSegments
% MNT: struct for electrode montage, see setElectrodeMontage
% OPT: property/value list or struct of options with fields/properties:
%  .scaleGroup -  groups of channels, where each group should
%                 get the same y limits, cell array of cells,
%                 default {scalpChannels, {'EMG*'},{'EOGh'},{'EOGv'}};
%                 As group you can also use scalpChannels (without quotes!)
%                 or 'all' (with quotes!).
%  .scalePolicy - says how the y limits are chosen:
%                 'auto': choose automatically,
%                 'sym': automatically but symmetric around 0
%                 'individual': choose automatically y limits individual
%                    for each channel
%                 'individual_sym': choose automatically y limits individual
%                    for each channel, but symmetric around 0
%                 'individual_tight': choose automatically y limits individual
%                    for each channel, but tight limits
%                 [lower upper]: define y limits;
%                 .scapePolicy is usually a cell array, where
%                 each cell corresponds to one .scaleGroup. Otherwise it
%                 applies only to the first group (scalpChannels by defaults).
%  .scaleUpperLimit - values (in magnitude) above this limit are not 
%                 considered, when choosing the y limits, default inf
%  .scaleLowerLimit - values (in magnitude) below this limit are not 
%                 considered, when choosing the y limits, default 0
%  .titleDir    - direction of figure title,
%                 'horizontal' (default), 'vertical', 'none'
%
%    * The following properties of OPT are passed to plotChannel:
%  .xUnit  - unit of x axis, default 'ms'
%  .yUnit  - unit of y axis, default epo.unit if this field
%            exists, '\muV' otherwise
%  .yDir   - 'normal' (negative down) or 'reverse' (negative up)
%  .refCol - color of patch indicating the baseline interval
%  .colorOrder -  defines the colors for drawing the curves of the
%                 different classes. if not given the colorOrder
%                 of the current axis is taken. as special gimmick
%                 you can use 'rainbow' as colorOrder.
%  .xGrid, ... -  many axis properties can be used in the usual
%                 way
%  .xZeroLine   - if true, draw an axis along the x-axis at y=0
%  .yZeroLine   - if true, draw an axis along the y-axis at x=0
%  .zeroLine*   - with * in {'Color','Style'} selects the
%                 drawing style of the axes at x=0/y=0
%  .axisTitle*  - with * in {'Color', 'HorizontalAlignment',
%                 'VerticalAlignment', 'FontWeight', 'FontSize'}
%                 selects the appearance of the subplot titles.
%
% SEE  makeEpochs, setDisplayMontage, plotChannel, grid_*

% Author(s): Benjamin Blankertz, Feb 2003 & Mar 2005

opt= propertylist2struct(varargin{:});
[opt,isdefault]= ...
    set_defaults(opt, ...
                 'yDir', 'normal', ...
                 'xUnit', 'ms', ...
                 'yUnit', 'dB', ...
                 'zUnit', 'Hz', ...
                 'tightenBorder', 0.03, ...
                 'axisType', 'box', ...
                 'box', 'on', ...
                 'shiftAxesUp', [], ...
                 'shrinkAxes', [1 1], ...
                 'oversizePlot', 1, ...
                 'scalePolicy', 'auto', ...
                 'scaleUpperLimit', inf, ...
                 'scaleLowerLimit', 0, ...
                 'legendVerticalAlignment', 'none', ...
                 'xTickAxes', '*', ...
                 'figureColor', [0.8 0.8 0.8], ...
                 'titleDir', 'horizontal', ...
                 'axisTitleHorizontalAlignment', 'center', ...
                 'axisTitleVerticalAlignment', 'top', ...
                 'axisTitleColor', [0 0 0], ...
                 'axisTitleFontSize', 12, ...
                 'axisTitleFontWeight', 'bold', ...
                 'scaleShowOrientation', 1, ...
                 'plotStd', 0, ...
                 'cLim', [], ...
                 'cLimPolicy', 'sym', ...
                 'colormap', jet(51));
             
if nargin<2 | isempty(mnt),
  mnt= strukt('clab',epo.clab);
else
  mnt= mnt_adaptMontage(mnt, epo);
end

if strcmpi(opt.axisType, 'cross'),  %% other default values for 'cross'
  opt= opt_overrideIfDefault(opt, isdefault, ...
                             'box', 'off', ...
                             'shrinkAxes', [0.9 0.9], ...
                             'axisTitleVerticalAlignment', 'cap');
%                             'oversizePlot', 1.5, ...
end
if opt.oversizePlot>1,
  [opt, isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'visible', 'off');
end
if isfield(opt, 'xTick'),
  [opt, isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'shrinkAxes', 0.8);
end
if isdefault.shiftAxesUp & ...
      (isfield(opt, 'xTick') & ~isempty(opt.xTick)), ...
      opt.shiftAxesUp= 0.05;
end
if isdefault.xUnit & isfield(epo, 'xUnit'),
  opt.xUnit= epo.xUnit;
end
if isdefault.yUnit & isfield(epo, 'yUnit'),
  opt.yUnit= epo.yUnit;
end
if isdefault.zUnit & isfield(epo, 'zUnit'),
  opt.zUnit= epo.zUnit;
end
dum = opt.zUnit;
opt.zUnit = opt.yUnit;
opt.yUnit = dum;

if isdefault.cLimPolicy & ismember(opt.zUnit, {'dB', 'power'})
  opt.cLimPolicy= 'tight';
end

if isfield(opt, 'yLim'),
  if ~isdefault.scalePolicy,
    warning('opt.yLim overrides opt.scalePolicy');
  end
  opt.scalePolicy= {opt.yLim};
  isdefault.scalePolicy= 0;
end
if ~iscell(opt.scalePolicy),
  opt.scalePolicy= {opt.scalePolicy};
end
if ~isfield(opt, 'scaleGroup'),
  grd_clab= getClabOfGrid(mnt);
  if strncmp(opt.scalePolicy, 'individual', length('individual')),
    opt.scaleGroup= grd_clab;
    pol= opt.scalePolicy(length('individual')+1:end);
    if isempty(pol),
      pol= 'auto';
    else
      if pol(1)=='_', pol(1)=[]; end
    end
    opt.scalePolicy= repmat({pol}, size(grd_clab));
  else
    scalp_idx= scalpChannels(epo);
    if isempty(scalp_idx),
      opt.scaleGroup= {intersect(grd_clab, epo.clab)};
    else
      scalp_idx= intersect(scalp_idx, chanind(epo, grd_clab));
      emgeog_idx= chanind(epo, 'EOGh','EOGv','EMG*');
      others_idx= setdiff(1:length(epo.clab), [scalp_idx emgeog_idx]);
      opt.scaleGroup= {epo.clab(scalp_idx), {'EMG*'}, {'EOGh'}, {'EOGv'}, ...
                       epo.clab(others_idx)};
      def_scalePolicy= {'auto', [-5 50], 'sym', 'auto', 'auto'};
      if isdefault.scalePolicy,
        opt.scalePolicy= def_scalePolicy;
      else
        memo= opt.scalePolicy;
        opt.scalePolicy= def_scalePolicy;
        opt.scalePolicy(1:length(memo))= memo;
      end
    end
  end
elseif isequal(opt.scaleGroup, 'all'),
  opt.scaleGroup= {getClabOfGrid(mnt)};
elseif ~iscell(opt.scaleGroup),
  opt.scaleGroup= {opt.scaleGroup};
end
if length(opt.scalePolicy)==1 & length(opt.scaleGroup)>1,
  opt.scalePolicy= repmat(opt.scalePolicy, 1, length(opt.scaleGroup));
end
if length(opt.shrinkAxes)==1,
  opt.shrinkAxes= [1 opt.shrinkAxes];
end

if isfield(mnt, 'box'),
  mnt.box_sz(1,:)= mnt.box_sz(1,:) * opt.shrinkAxes(1);
  mnt.box_sz(2,:)= mnt.box_sz(2,:) * opt.shrinkAxes(2) * opt.oversizePlot;
  if isfield(mnt, 'scale_box_sz'),
   mnt.scale_box_sz(1)= mnt.scale_box_sz(1)*opt.shrinkAxes(1);
   mnt.scale_box_sz(2)= mnt.scale_box_sz(2)*opt.shrinkAxes(2)*opt.oversizePlot;
  end
end

if max(sum(epo.y,2))>1,
  epo= proc_average(epo, 'std',opt.plotStd);
end

if isempty(opt.cLim)
    if isequal(opt.cLimPolicy, 'tight')
        opt.cLim = [min(min(min(min(epo.x)))) max(max(max(max(epo.x))))];
    else 
        ma = max(abs([min(min(min(min(epo.x)))) max(max(max(max(epo.x))))]));
        opt.cLim = [-ma ma];
    end
end

for icl = 1:size(epo.y, 1)

figure;clf;set(gcf, 'color',opt.figureColor);

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end
nDisps= length(dispChans);
%% mnt.clab{dispChans(ii)} may differ from epo.clab{ii}, e.g. the former
%% may be 'C3' while the latter is 'C3 lap'
idx= chanind(epo, mnt.clab(dispChans));
axesTitle= apply_cellwise(epo.clab(idx), 'sprintf');

%w_cm= warning('query', 'bci:missing_channels');
%warning('off', 'bci:missing_channels');
%all_idx= 1:length(mnt.clab);
yLim= zeros(length(opt.scaleGroup), 2);
for ig= 1:length(opt.scaleGroup),
  ax_idx= chanind(mnt.clab(dispChans), opt.scaleGroup{ig});
  if isempty(ax_idx), continue; end
%  ch_idx= find(ismember(all_idx, ax_idx));
  ch_idx= dispChans(ax_idx);
  if isnumeric(opt.scalePolicy{ig}),
    yLim(ig,:)= opt.scalePolicy{ig};
  else
    dd= epo.x(:,:, ch_idx,:);
%    idx= find(~isinf(dd(:)));
    idx= find(abs(dd(:))<opt.scaleUpperLimit & ...
              abs(dd(:))>=opt.scaleLowerLimit);
    yl= [nanmin(dd(idx)) nanmax(dd(idx))];
    %% add border not to make it too tight:
    yl= yl + [-1 1]*opt.tightenBorder*diff(yl);
    if strncmp(opt.scalePolicy{ig},'tight',5),
      yLim(ig,:)= yl;
    else
      %% determine nicer limits
      dig= floor(log10(diff(yl)));
      if diff(yl)>1,
        dig= max(1, dig);
      end
      yLim(ig,:)= [trunc(yl(1),-dig+1,'floor') trunc(yl(2),-dig+1,'ceil')];
    end
  end
  if isequal(opt.scalePolicy{ig},'sym'),
    yl= max(abs(yLim(ig,:)));
    yLim(ig,:)= [-yl yl];
  end
  if any(isnan(yLim(ig,:))),
    yLim(ig,:)= [-1 1];
  end
  if ig==1 & length(ax_idx)>1,
%    scale_with_group1= setdiff(1:nDisps, chanind(mnt.clab(dispChans), ...
%                                                 [opt.scaleGroup{2:end}]));
%    set(H.ax(scale_with_group1), 'yLim',yLim(ig,:));
    ch2group= ones(1,nDisps);
  else
%    set(H.ax(ax_idx), 'yLim',yLim(ig,:));
    ch2group(ax_idx)= ig;
    for ia= ax_idx,
      if max(abs(yLim(ig,:)))>=100,
        dig= 0;
      elseif max(abs(yLim(ig,:)))>=1,
        dig= 1;
      else
        dig= 2;
      end
      axesTitle{ia}= sprintf('%s  [%g %g] %s', ...
                             axesTitle{ia}, ...
                             trunc(yLim(ig,:), dig), opt.yUnit);
    end
  end
end
%warning(w_cm);

opt_plot= {'legend',0, 'title','', 'unitDispPolicy','none', ...
           'grid_over_patches',0};
if isfield(mnt, 'box') & isnan(mnt.box(1,end)),
  %% no grid position for legend available
  opt_plot{2}= 0;
end

H(icl).ax= zeros(1, nDisps);
for ia= 1:nDisps,
  ic= dispChans(ia);
  H(icl).ax(ia)= axes('position', getAxisGridPos(mnt, ic));
  H(icl).chan(ia)= plotChannel_specgram(epo, icl, mnt.clab{ic}, opt, opt_plot{:}, ...
                          'yLim', yLim(ch2group(ia),:), ...
                          'axisTitle', axesTitle{ia}, 'title',0, ...
                          'smallSetup',1, 'cLim', opt.cLim, 'colormap', opt.colormap);
%   if ic==dispChans(1),
%     opt_plot{2}= 0;
%     H(icl).leg= H(icl).chan(ia).leg;
%     leg_pos= getAxisGridPos(mnt, 0);
%     if ~any(isnan(leg_pos)),
%       leg_pos_orig= get(H(icl).leg, 'position');
%       if leg_pos(4)>leg_pos_orig(4),
%         switch(lower(opt.legendVerticalAlignment)), 
%           case 'top',
%            leg_pos(2)= leg_pos(2)-leg_pos_orig(4)+leg_pos(4);
%          case 'middle',
%            leg_pos(2)= leg_pos(2)+(leg_pos(4)-leg_pos_orig(4))/2;
%         end
%       end
%       leg_pos(3:4)= leg_pos_orig(3:4);  %% use original size
%       set(H(icl).leg, 'position', leg_pos);
%       ud= get(H(icl).leg, 'userData');
%       ud= set_defaults(ud, 'type','ERP plus', 'chan','legend');
%       set(H(icl).leg, 'userData',ud);
%       if exist('verLessThan')~=2 | verLessThan('matlab','7'),
%         set(H(icl).leg, 'visible','off');
%         set(get(H(icl).leg,'children'), 'visible','on');
%       end
%     end
%   end
end

if isfield(mnt, 'scale_box') & all(~isnan(mnt.scale_box)),
  ax_idx= chanind(mnt.clab(dispChans), opt.scaleGroup{1});
  axes(H(icl).ax(ax_idx(1)));
  H(icl).scale= grid_addScale(mnt, opt);
end
grid_over_patches('axes',H(icl).ax);

if ~isdefault.xTickAxes,
  h_xta= H(icl).ax(chanind(mnt.clab(dispChans), opt.xTickAxes));
  set(setdiff(H(icl).ax, h_xta), 'XTickLabel','');
end

if ~strcmp(opt.titleDir, 'none'),
  tit= '';
  if isfield(opt, 'title'),
    tit= [opt.title ':  '];
  elseif isfield(epo, 'title'),
    tit= [untex(epo.title) ':  '];
  end
  if isfield(epo, 'className'),
    tit= [tit, epo.className{icl} ', '];
  end
  if isfield(epo, 'N'),
    tit= [tit, 'N= ' num2str(epo.N(icl)) ',  '];
  end
  if isfield(epo, 't'),
    tit= [tit, sprintf('[%g %g] %s  ', trunc(epo.t([1 end])), opt.xUnit)];
  end
  if isfield(epo, 'f'),
    tit= [tit, sprintf('[%g %g] %s  ', round(epo.f([1 end])), opt.yUnit)];
  end
  tit= [tit, sprintf('[%g %g] %s', trunc(opt.cLim, 2), opt.zUnit)];
  if strcmpi(opt.yDir, 'reverse'),
    tit= [tit, ' neg. up'];
  end
  if isfield(opt, 'titleAppendix'),
    tit= [tit, ', ' opt.titleAppendix];
  end
  H(icl).title= addTitle(tit, opt.titleDir);
end

if ~isempty(opt.shiftAxesUp) & opt.shiftAxesUp~=0,
  shiftAxesUp(opt.shiftAxesUp);
end

end

if nargout==0,
  clear H;
end

