function H= drawScalpOutline(mnt, varargin)

% DRAWSCALPOUTLINE - schematic sketch of the scalp (head, nose, ears) and
% the electrode positions including labels
%
% Synopsis:
%   H = DRAWSCALPOUTLINE(MNT, <OPT>)
%
% Arguments:
%   MNT: mount struct
%
%   OPT: struct or property/value list of optional properties:
%   'showLabels'    : if true, electrode labels are depicted in circles,
%                     otherwise only the positions are indicated; default 0.
%   'labelProps'      : text properties of labels (eg 'FontSize', 'Color')
%   'minorLabelProps' : text properties of long labels (>3 characters)
%   'markerProps'     : line/marker properties of electrode position markers
%   'lineProps'       : specify line properties of scalp outline
%   'markChans'       : give cell array of channels to be marked
%   'markLabelProps'  : same as labelProperties, but only for marked
%                            channels
%   'markMarkerProps' : same as markerProperties, but only for marked
%                            channels
%   'dispChans'       : channels to display, as a cell array of labels
%                       or alternatively a vector of indices
%   'ears'            : if true, draws ears; default 0
%   'reference'       : draws the reference at 'nose' 'linked_ears'
%                        or 'linked_mastoids'; default none
%   'referenceProps'  : reference text properties
%   'ax'              : provide axes handle (useful if you want to plot 
%                       the scalp in a subplot)
%
%   The following OPT properties are obsolete and should not be used any
%   more. They are still supported for compatibility reasons:
%   'crossSize','crossWidth','markerSize','fontSize'
%   'minorFontSize','linespec','mark_channels','mark_properties'
%
%Output:
%   H    : struct with handles for the current, the scalp plot, 
%
% Example:
% drawScalpOutline(mnt,'showLabels',1,'lineProps',{'LineWidth',3},'labelProps',{'FontWeight','bold'},'markerProps',{'LineWidth',5,'Color','red'},'ears',1); 
%
% This plots a scalp with ears and red-colored markers.
%
% Author: Benjamin Blankertz
%
% 2009-06-1  Matthias Treder: 
%          added ears and reference, rearranged and extended OPT struct so
%          all usual line/font/marker options can be used
% 2009-06-10 Matthias Treder:
%          Added backward-compatibility
%
%
%% Normalized units (for positioning, set back later)
oldUnit = get(gcf,'units');
set(gcf,'units','normalized');

%% Process optional commands
opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'showLabels', 0, ...
                 'labelProps', {'FontSize',8}, ...
                 'minorLabelProps', {'FontSize',6}, ...
                 'lineProps', {'Color' 'k'}, ...
                 'markChans', [], ...
                 'markLabelProps', {'FontSize',12,'FontWeight','bold'}, ...
                 'markMarkerProps', {'LineWidth',3, 'MarkerSize',22}, ...
                 'dispChans', 1:length(mnt.clab), ...
                 'ears',0,...
                 'reference',0,...
                 'offset',[0 0], ...
                 'referenceProps',{'FontSize',8,'FontWeight','bold','BackgroundColor',[.8 .8 .8],'HorizontalAlignment','center','Margin',2},...
                 'ax',gca,...
                 'H',struct('ax',NaN) ...
                 ); 

if opt.showLabels
    [opt, isdefault]= set_defaults(opt, ...
        'markerProps', {'Marker','o','MarkerSize',20,'MarkerEdgeColor','k'});
else % show cross
    [opt, isdefault]= set_defaults(opt, ...
        'markerProps', {'Marker','+','MarkerSize',2,'LineWidth',.2,'MarkerEdgeColor','k'});
end;
             
%% Check for spelling mistakes
if isfield(opt,'markChannels')
    opt.markChans = opt.markChannels;
    opt = rmfield(opt,'markChannels');
    bbci_warning('Please use ''markChans'' instead of ''markChannels''', ...
                 'drawscalp', mfilename);
end
%% When showLabels == 0: set default marker 
%% Check for obsolete options
fn = fieldnames(opt);
oldfields = {'crossSize' 'crossWidth' 'markerSize' ...
    'fontSize' 'minorFontSize' 'linespec' 'mark_channels' ...
    'mark_properties'};
% Check whether any of the obsolete options has been set
obsolete = find(ismember(fn,oldfields));
if obsolete
    oldn = fn(obsolete);
%    bbci_warning(sprintf('The use of %s is deprecate',vec2str(oldn)), ...
%                 'drawscalp', mfilename)
    if ~opt.showLabels && (isfield(opt,'crossSize') ||isfield(opt,'crossWidth'))
        if isfield(opt,'crossSize')
            cs = opt.crossSize;
        else cs = 2; end
        if isfield(opt,'crossWidth')
            cw = opt.crossWidth;
        else cw = .2; end
        % Set '+' as marker 
        opt.markerProps = {'Marker' '+' ...
            'MarkerSize' cs 'LineWidth' cw 'Color' 'k'};
    end
    if isfield(opt,'markerSize')
        opt.markerProps = {opt.markerProps{:} 'MarkerSize' opt.markerSize};
    end
    if isfield(opt,'fontSize')
        opt.labelProps = {opt.labelProps{:} 'FontSize' opt.fontSize};
    end
    if isfield(opt,'minorFontSize')
        opt.minorLabelProps = {opt.minorLabelProps{:} 'FontSize' opt.minorFontSize};
    end
    if isfield(opt,'linespec')
        if length(opt.linespec)==1,
          opt.linespec= cat(1, {'Color'}, opt.linespec);
        end
        if ~isfield(opt, 'lineProps')
            opt.lineProps = opt.linespec;
        else
            opt.lineProps = {opt.lineProps{:}, opt.linespec{:}};
%           bbci_warning('Both opt.lineProps and opt.linespec are set. Now using merged version', 'propMerge', mfilename);
        end
    end
    if isfield(opt,'mark_channels')
        opt.markChans = opt.mark_channels;
    end
    if isfield(opt,'mark_properties')
        opt.markMarkerProps = opt.mark_properties;
    end
    % Remove all obsolete fields
    for k=1:numel(obsolete)
        opt = rmfield(opt,oldn{k});
    end
end

%% Check for missing properties
% If no other marker was set, set default marker 'o'
if ~any(strcmpi('marker',opt.markerProps)),
    opt.markerProps = {opt.markerProps{:},'Marker','o'};
end
% If size is not set, set size to 20
if ~any(strcmpi('markersize',opt.markerProps)),
    opt.markerProps = {opt.markerProps{:},'MarkerSize',20};
end
% If no color for scalp was set, set default color black
if ~any(strcmpi('color',opt.lineProps)),
    opt.lineProps = {opt.lineProps{:}, 'Color','k'};
end

% If channels are given as labels in cells, turn into indices
if iscell(opt.dispChans)
    opt.dispChans = chanind(mnt.clab, opt.dispChans);
end

%% Get axes handle and old position
H= opt.H;
H.ax = opt.ax;
set(gcf,'CurrentAxes',H.ax);
old_pos = get(gca,'position');
%% Get coordinates
xe= mnt.x(opt.dispChans)+opt.offset(1);
ye= mnt.y(opt.dispChans)+opt.offset(2);
%% Plot head
T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
H.head= plot(xx+opt.offset(1), yy+opt.offset(2), opt.lineProps{:});
hold on;
%% Plot ears
if opt.ears
    earw = .06; earh = .2;
    H.ears(1)= plot(xx*earw-1-earw+opt.offset(1), yy*earh+opt.offset(2), opt.lineProps{:});
    H.ears(2)= plot(xx*earw+1+earw+opt.offset(1), yy*earh+opt.offset(2), opt.lineProps{:});
end
%% Plot nose
nose= [1 1.1 1];
nosi= [86 90 94]+1;
H.nose= plot(nose.*xx(nosi)+opt.offset(1), nose.*yy(nosi)+opt.offset(2), opt.lineProps{:});
%% Add reference
if opt.reference
    ref = ' REF ';
%    H.ref(1) = text(0,0,ref,opt.referenceProps{:});
    switch(opt.reference),
        case 'nose'
            noseroot = min(nose.*yy(nosi));
            H.ref(1) = text(0,0,ref,opt.referenceProps{:},...
                'HorizontalAlignment','Center',...
                'VerticalAlignment','top', 'position',[0 noseroot]);
        case 'linked_ears'
            if exist('earw','var') && exist('earh','var')
                xear = max(get(H.ears(2),'XData'));
                year = -earh/2;
            else xear = max(xx); year=0;
            end
            H.ref(1) = text(-xear,year,ref,opt.referenceProps{:},...
                'HorizontalAlignment','right');
            H.ref(2) = text(xear,year,ref,opt.referenceProps{:},...
                'HorizontalAlignment','left');
            set(H.ref(:),'VerticalAlignment','middle');
        case 'linked_mastoids'
            if exist('earw','var') && exist('earh','var')
                year = -earh*1.2;
            else year=0;
            end
            H.ref(1) = text(-1,year,ref,opt.referenceProps{:},...
                'HorizontalAlignment','right');
            H.ref(2) = text(1,year,ref,opt.referenceProps{:},...
                'HorizontalAlignment','left');
            set(H.ref(:),'VerticalAlignment','middle');
            set(H.ref(:),'VerticalAlignment','top');
    end
end
%% Add markers & labels
opt.markChans= chanind(mnt.clab(opt.dispChans), opt.markChans);
% Plot markers
H.label_markers = [];
for k=1:numel(xe)
    H.label_markers(k)= plot(xe(k), ye(k),'LineStyle','none',opt.markerProps{:});
    hold on
end
% Mark marked markers
if ~isempty(opt.markChans)
    set(H.label_markers(opt.markChans), opt.markMarkerProps{:});
end
% Plot labels
if opt.showLabels,
  labs= {mnt.clab{opt.dispChans}};
  H.label_text= text(xe, ye, labs);
  set(H.label_text, 'horizontalAlignment','center',opt.labelProps{:});
  % Find labels with >3 letters and set their properties
  strLen= apply_cellwise(labs, 'length');
  strLen = [strLen{:}];
  iLong= strLen>3;
  set(H.label_text(iLong), opt.minorLabelProps{:});
  if ~isempty(opt.markChans),
    set(H.label_text(opt.markChans), opt.markLabelProps{:});
  end
end

%box off;
hold off;
set(H.ax, 'xTick',[], 'yTick',[]); %, 'xColor','w', 'yColor','w');
axis('xy', 'tight', 'equal', 'tight');
%% relax XLim, YLim:
xLim= get(H.ax, 'XLim');
yLim= get(H.ax, 'YLim');
set(H.ax, 'XLim',xLim+[-1 1]*0.03*diff(xLim), ...
      'YLim',yLim+[-1 1]*0.03*diff(yLim));
%% Set back old figure units
set(gcf,'units',oldUnit);
