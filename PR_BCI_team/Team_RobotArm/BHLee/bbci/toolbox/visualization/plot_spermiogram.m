function ha= plot_spermiogram(file_dscr, file_dtct, varargin)
%ha= plot_spermiogram(file_dscr, file_dtct, opt [struct/propertylist])
%
% IN  file_dscr - file name of discrimination traces, as produced
%                 by plot_tubes_discrimination
%     file_dtct - file name of detection traces, as produced
%                 by plot_tubes_detection
%     opt         struct or/and propertylist
%        .ival
%        .mark
%        .smooth - apply moving average to traces
%        .xScale
%        .yScale
%        .xLim
%        .yLim

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'ival', [-350 -100], ...
                  'mark', -150, ...
                  'smooth', 0, ...
                  'xScale', 1, ...
                  'yScale', 1, ...
                  'xLim', 1.6 * [-1 1], ...
                  'yLim', [-1.4 1.8], ...
		  'subplots', 4, ...
		  'stopAtBorder', 0, ...
		  'idx',[]);

load(file_dtct);
traces_dtct= outTraces;
E_dtct= dsply.E;
load(file_dscr);
traces_dscr= outTraces;
E_dscr= dsply.E;

sz= size(traces_dscr);
if length(sz)>2 & sz(1)>1,
  warning('dscr: rather use loo than xval');
  traces_dscr= squeeze(traces_dscr(1,:,:));
else
  traces_dscr= squeeze(traces_dscr);
end
sz= size(traces_dtct);
if length(sz)>2 & sz(1)>1,
  warning('dtct: rather use loo than xval');
  traces_dtct= squeeze(traces_dtct(1,:,:));
else
  traces_dtct= squeeze(traces_dtct);
end

if opt.smooth>1,
  traces_dscr= movingAverageCausal(traces_dscr, opt.smooth);
  traces_dtct= movingAverageCausal(traces_dtct, opt.smooth);
end

if opt.stopAtBorder,
  traces_dscr= max(opt.xLim(1), traces_dscr);
  traces_dscr= min(opt.xLim(2), traces_dscr);
  traces_dtct= max(opt.xLim(1), traces_dtct);
  traces_dtct= min(opt.xLim(2), traces_dtct);
end
  
E= intersect(E_dscr, E_dtct);
idx= find(ismember(E_dscr, E_dtct));
traces_dscr= traces_dscr(idx,:);
idx= find(ismember(E_dtct, E_dscr));
traces_dtct= traces_dtct(idx,:);

traces_dscr= opt.xScale*traces_dscr;
traces_dtct= opt.yScale*traces_dtct;

nEvents= size(traces_dscr, 2);
if nEvents~=size(traces_dscr),
  error('inconsistent number of events');
end
if isempty(opt.idx),
  opt.idx= 1:nEvents;
end
nSel= length(opt.idx);

iv= max(find(E<=opt.ival(1))):min(find(E>=opt.mark));
ivm= max(find(E<=opt.mark)):min(find(E>=opt.ival(2)));

col_h= labels(2,:)/3;

clf;
for ip= 1:opt.subplots,
  evt= ceil(nSel/opt.subplots*(ip-1)+1):ceil(nSel/opt.subplots*ip);
  ha(ip)= suplot(opt.subplots, ip, [0.05 0.05], [0.03 0.03]);
  
  hold on;
  line([-10 10; 0 0]', [0 0; -100 100]', 'color','k');
  for ie= opt.idx(evt),
    plot(traces_dscr(iv,ie), traces_dtct(iv,ie), ...
         'color',hsv2rgb([col_h(ie) 0.25 1]));
  end
  for ie= opt.idx(evt),
    plot(traces_dscr(ivm,ie), traces_dtct(ivm,ie), ...
         'color',hsv2rgb([col_h(ie) 0.6 0.8]));
    plot(traces_dscr(ivm(end),ie), traces_dtct(ivm(end),ie), '.', ...
         'color',hsv2rgb([col_h(ie) 0.8 0.8]));
  end
end
set(ha, 'xLim',opt.xLim, 'yLim',opt.yLim, 'box','on');
