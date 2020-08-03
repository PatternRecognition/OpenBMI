function [ht, ax, hleg, he]= plotERPgrid(epo, mnt, OPT, varargin)
%[ht, ax]= plotERPgrid(epo, mnt, <OPT>)
%
% IN   epo     - struct of epoched signals, see makeSegments
%      mnt     - struct for electrode montage, see setElectrodeMontage
%      OPT
%           .xUnit  - unit of x axis, default 'ms'
%           .yUnit  - unit of y axis, default '\\muV'
%           .refCol - color of patch indicating the baseline interval
%           .titleDir - direction of figure title,
%                       'horizontal', 'vertical', 'none'
%           .scaleChannelgroup - 
%
% OUT  ht      - handle of title string
%      ax      - handle of subaxes
%
% SEE  showERP, makeSegments, setElectrodeMontage

% bb, FhG-FIRST 04/02

bbci_obsolete(mfilename, 'grid_plot');

if ~exist('OPT', 'var'), OPT.yDir='normal'; end
if ~isfield(OPT, 'refCol'), OPT.refCol= 0.8; end
if ~isfield(OPT, 'xUnit'), OPT.xUnit= 'ms'; end
if ~isfield(OPT, 'yUnit'), OPT.yUnit= '\muV'; end
if ~isfield(OPT, 'xGrid'), OPT.xGrid= 'on'; end
if ~isfield(OPT, 'yGrid'), OPT.yGrid= 'on'; end
if ~isfield(OPT, 'xTickLabel'), OPT.xTickLabel=[]; end
if isequal(OPT.xTickLabel,'auto'), OPT= rmfield(OPT, 'xTickLabel'); end
if ~isfield(OPT, 'yTickLabel'), OPT.yTickLabel= []; end
if isequal(OPT.yTickLabel,'auto'), OPT= rmfield(OPT, 'yTickLabel'); end
if ~isfield(OPT, 'xZeroLine'), OPT.xZeroLine= 'on'; end
if ~isfield(OPT, 'yZeroLine'), OPT.yZeroLine= 'on'; end
if ~isfield(OPT, 'titleDir'), OPT.titleDir='horizontal'; end
if ~isfield(OPT, 'scaleGroup'),
  OPT.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};
end
if ~isfield(OPT, 'scalePolicy'),
  OPT.scalePolicy= {'none', [-5 50], 'none'};
end
if ischar(OPT.scalePolicy),
  OPT.scalePolicy= repmat({OPT.scalePolicy}, 1, length(OPT.scaleGroup));
end


clf;
%set(gcf, 'color',[1 1 1]);
flags= {'legend', 'grid', varargin{:}};

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end

style_list= {'xGrid', 'xTick', 'xTickLabel', 'xLim', ...
             'yGrid', 'yTick', 'yTickLabel', 'yLim', 'yDir', ...
             'colorOrder', 'lineStyleOrder', ...
             'lineWidth', 'tickLength', ...
             'fontName', 'fontSize', 'fontUnits', 'fontWeight'};
ai= 0;
axesStyle= {};
opt_fields= fieldnames(OPT);
for is= 1:length(style_list),
  sm= strmatch(lower(style_list{is}), lower(opt_fields), 'exact');
  if length(sm)==1,
    ai= ai+2;
    axesStyle(ai-1:ai)= {style_list{is}, getfield(OPT, opt_fields{sm})};
  end
end

nDisps= length(dispChans);
ax= zeros(1, nDisps);
axesTitle= cell(1, nDisps);
for ia= 1:nDisps,
  ic= dispChans(ia);
  ax(ia)= axes('position', getAxisGridPos(mnt, ic));
  set(ax(ia), axesStyle{:});
  [nEvents, dummy, hc]= showERP(epo, mnt, mnt.clab{ic}, flags{:});
  set(ax(ia), axesStyle{:});
  if ic==dispChans(1),
%    set(hc, 'position', getAxisGridPos(mnt, 0));
    flags= {flags{2:end}}; 
    hleg= hc;
  end
%  set(ax(ia), 'position', getAxisHeadPos(mnt, ic, axisSize));
%  axis off;
  axesTitle{ia}= mnt.clab{ic};
end

%all_idx= 1:length(mnt.clab);
yLim= zeros(length(OPT.scaleGroup), 2);
for ig= 1:length(OPT.scaleGroup),
  ax_idx= chanind(mnt.clab(dispChans), OPT.scaleGroup{ig});
%  ch_idx= find(ismember(all_idx, ax_idx));
  ch_idx= dispChans(ax_idx);
  if isnumeric(OPT.scalePolicy{ig}),
    yLim(ig,:)= OPT.scalePolicy{ig};
  else
    yLim(ig,:)= unifyYLim(ax(ax_idx));
  end
  if isequal(OPT.scalePolicy{ig},'sym'),
    yl= max(abs(yLim(ig,:)));
    yLim(ig,:)= [-yl yl];
  end
  if ig==1,
    scale_with_group1= setdiff(1:nDisps, chanind(mnt.clab(dispChans), ...
                                                 [OPT.scaleGroup{2:end}]));
    set(ax(scale_with_group1), 'yLim',yLim(ig,:));
    ch2group= ones(1,nDisps);
  else
    set(ax(ax_idx), 'yLim',yLim(ig,:));
    ch2group(ax_idx)= ig;
    for ia= ax_idx,
      if max(abs(yLim(ig,:)))>=100,
        dig= 0;
      else
        dig= 1;
      end
      axesTitle{ia}= sprintf('%s  [%g %g] %s', ...
                             axesTitle{ia}, ...
                             trunc(yLim(ig,:), dig), OPT.yUnit);
    end
  end
end


he= [];
for ia= 1:nDisps,
  ic= dispChans(ia);
  axes(ax(ia));
  xLim= get(gca, 'xLim');
  x= xLim(1)+0.05*diff(xLim);
%  x= xLim(2)-0.05*diff(xLim);
  yl= yLim(ch2group(ia),:);
  y= yl(2-strcmpi(OPT.yDir, 'reverse'));
  he= [he text(x, y, axesTitle{ia})];
  if strcmpi(OPT.xZeroLine, 'on'),
    hl= line(xLim, [0 0], 'color','k');
    moveObjectBack(hl);
  end
  if strcmpi(OPT.yZeroLine, 'on'),
    hl= line([0 0], yl, 'color','k');
    moveObjectBack(hl);
  end
  if isfield(epo, 'refIval'),
    yPatch= [-0.05 0.05] * diff(yl);
    if length(OPT.refCol)==1,
      refCol= OPT.refCol*[1 1 1];
    else
      refCol= OPT.refCol;
    end
    hp= patch(epo.refIval([1 2 2 1]), yPatch([1 1 2 2]), refCol);
    set(hp, 'edgeColor','none');
    moveObjectBack(hp);
  end
end
set(he, 'verticalAlignment','top');
%set(he, 'horizontalAlignment','right');



if strcmp(OPT.titleDir, 'none'),
  return;
end

tit= '';
if isfield(epo, 'title'),
  tit= [untex(epo.title) ':  '];
end
if isfield(epo, 'className'),
  tit= [tit vec2str(epo.className, [], ' / ') ', '];
end
tit= [tit 'N= ' vec2str(nEvents,[],'/') ',  '];
if isfield(epo, 't'),
  tit= [tit sprintf('[%g %g] %s  ', trunc(epo.t([1 end])), OPT.xUnit)];
end
tit= [tit sprintf('[%g %g] %s', trunc(yLim(1,:)), OPT.yUnit)];
ht= addTitle(tit, OPT.titleDir);
