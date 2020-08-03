function H= grid_image(epo, mnt, varargin)
%GRID_IMAGE - Grid layout of 2D data per channel
%
%Synopsis:
% H= grid_image(EPO, MNT, <OPT>)
%
%Input:
% EPO: struct of epoched signals, see makeSegments
% MNT: struct for electrode montage, see setElectrodeMontage
% OPT: property/value list or struct of options with fields/properties:
%  .clim
%  .climmode
%  .colormap
%  .units_at_clab

% Author(s): Benjamin Blankertz

opt= propertylist2struct(varargin{:});
[opt,isdefault]= ...
    set_defaults(opt, ...
                 'yDir', 'normal', ...
                 'xUnit', 'ms', ...
                 'yUnit', '\muV', ...
								 'climmode','sym', ...
								 'colormap', cmap_posneg(51), ...
                 'shiftAxesUp', 0.05, ...
                 'shiftAxesRight', 0.04, ...
                 'shrinkAxes', [0.97 0.97], ...
                 'figureColor', [0.8 0.8 0.8], ...
                 'titleDir', 'horizontal', ...
                 'axisTitleHorizontalAlignment', 'center', ...
                 'axisTitleVerticalAlignment', 'top', ...
                 'axisTitleColor', 'k', ...
                 'axisTitleFontSize', get(gca,'fontSize'), ...
                 'axisTitleFontWeight', 'normal', ...
								 'units_at_clab', 'auto');

if nargin<2 | isempty(mnt),
  mnt= strukt('clab',epo.clab);
else
  mnt= mnt_adaptMontage(mnt, epo);
end

if length(opt.shrinkAxes)==1,
  opt.shrinkAxes= [1 opt.shrinkAxes];
end
if isdefault.xUnit & isfield(epo, 'xUnit'),
  opt.xUnit= epo.xUnit;
end
if isdefault.yUnit & isfield(epo, 'yUnit'),
  opt.yUnit= epo.yUnit;
end
if isdefault.climmode & isfield(opt, 'clim'),
	opt.climmode= 'manual';
end
if isfield(mnt, 'box'),
  mnt.box_sz(1,:)= mnt.box_sz(1,:) * opt.shrinkAxes(1);
  mnt.box_sz(2,:)= mnt.box_sz(2,:) * opt.shrinkAxes(2);
  if isfield(mnt, 'scale_box_sz'),
   mnt.scale_box_sz(1)= mnt.scale_box_sz(1)*opt.shrinkAxes(1);
   mnt.scale_box_sz(2)= mnt.scale_box_sz(2)*opt.shrinkAxes(2);
  end
end

if isequal(opt.units_at_clab, 'auto'),
	ilow= find(mnt.box(2,:)==min(mnt.box(2,:)));
	[mi,mm]= min(mnt.box(1,ilow));
	opt.units_at_clab= ilow(mm);
else
	opt.units_at_clab= chanind(mnt, opt.units_at_clab);
end

if max(sum(epo.y,2))>1,
  epo= proc_average(epo);
end

clf;
set(gcf, 'color',opt.figureColor);

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end
nDisps= length(dispChans);
%% mnt.clab{dispChans(ii)} may differ from epo.clab{ii}, e.g. the former
%% may be 'C3' while the latter is 'C3 lap'
idx= chanind(epo, mnt.clab(dispChans));
axesTitle= apply_cellwise(epo.clab(idx), 'sprintf');

xx= epo.x(:,idx);
mi= min(xx(:));
ma= max(xx(:));
mm= max(abs([mi ma]));
switch(opt.climmode),
 case 'sym',
	cLim= [-mm mm];
 case '0tomax',
	cLim= [0 mm];
 case 'range',
	cLim= [mi ma];
 case 'manual',
	cLim= opt.clim;
end

set(gcf, 'Colormap',opt.colormap);
sz= size(epo.x);
xx= reshape(epo.x, [epo.dim sz(2:end)]);
for ia= 1:nDisps,
  ic= dispChans(ia);
  H.ax(ia)= axes('position', getAxisGridPos(mnt, ic));
  H.chan(ia)= imagesc(epo.t{1}, epo.t{2}, xx(:,:,ic)');
	set(gca, 'YDir',opt.yDir, 'CLim',cLim);
	axis_title(axesTitle(ia), 'vpos',0.98, 'fontWeight','bold', ...
						 'verticalAlignment','top');
	if ic~=opt.units_at_clab,
		set(gca, 'XTickLabel',[], 'YTickLabel',[]);
	end
%	if ia==1,
%    H.ax_cb= axes('position', getAxisGridPos(mnt, 0));
%		imagesc(linspace(cLim(1), cLim(2), 100)');
%		set(gca, 'XTick',[]);
%  end
end

if ~strcmp(opt.titleDir, 'none'),
  tit= '';
  if isfield(opt, 'title'),
    tit= [opt.title ':  '];
  elseif isfield(epo, 'title'),
    tit= [untex(epo.title) ':  '];
  end
  if isfield(epo, 'className'),
    tit= [tit, vec2str(epo.className, [], ' / ') ', '];
  end
  if isfield(epo, 'N'),
    tit= [tit, 'N= ' vec2str(epo.N,[],'/') ',  '];
  end
  if isfield(epo, 't'),
    tit= [tit, sprintf('[%g %g] %s,  ', ...
                       trunc(epo.t{1}([1 end])), opt.xUnit{1})];
    tit= [tit, sprintf('[%g %g] %s,  ', ...
                       trunc(epo.t{2}([1 end])), opt.xUnit{2})];
  end
  tit= [tit, sprintf('[%g %g] %s', trunc(cLim(1,:)), opt.yUnit)];
  if isfield(opt, 'titleAppendix'),
    tit= [tit, ', ' opt.titleAppendix];
  end
  H.title= addTitle(tit, opt.titleDir);
end

if ~isempty(opt.shiftAxesUp) & opt.shiftAxesUp~=0,
  shiftAxesUp(opt.shiftAxesUp);
end
if ~isempty(opt.shiftAxesRight) & opt.shiftAxesRight~=0,
  shiftAxesRight(opt.shiftAxesRight);
end

if nargout==0,
  clear H;
end
