function H = plotChannelsOnScalp(dat, mnt, varargin)
%PLOTSCALPERP - Superimposes 2D or 3D channel plots topographically
%on a scalp drawing.
%
%Usage:
% H = PLOTCHANNELSONSCALP(DAT, MNT, <OPT>)
%
%Input:
% DAT  - struct of epoched (and possibly averaged) EEG data. 
% MNT  - struct defining an electrode montage
% <OPT> - struct or property/value list of optional fields/properties:
%
%   size      : Normalized size (eg [.01 .02]) of each plot; can also be a
%               cell array with a size vector for each electrode
%   dispChans : cell array of channels to be displayed
%   scalePlots: plots are scaled according to a function; 
%               'xlin' linear scaling along x axis (starting from midline)
%               'ylin' linear scaling along x axis (starting from midline)
%               'radial' radial linear scaling starting from midpoint
%   scaleMax  : most extreme size of scaled plot (default: size/2)
%   axis      : turn axes of plots on or off
%   stretch   : stretches position by changing the distance from the
%               electrode site to center by a strech factor. Note that
%               scaling is performed based on the old positions.
%   shift     : shifts position of ALL electrode sites by [x y] in
%               normalized electrode coordinates; default [0 0]
%
%Output:
% H:     Handle to several graphical objects.
%
%Note:
% Calls drawScalpOutline and plotChannel. See the corresponding files for 
% additional options. Uses also axescoord2figurecoord.

%% Options and defaults
H = struct();
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...          % plotChannelsOnScalp defaults
                  'size',[.05 .05],...
                  'scalePlots',0,...
                  'axis','on',...
                  'stretch',0,...
                  'shift',[0 0]);
opt= set_defaults(opt, ...          % drawScalpOutline defaults
                  'markerProps',{'Marker','.','MarkerSize',6, 'Color',[1 1 1]});  % make marker invisible
opt= set_defaults(opt, ...          % plotChannel defaults
                  'xLim',[dat.t(1) dat.t(end)],...
                  'xGrid','off','yGrid','off',...
                  'xUnit','','yUnit','',...
                  'xTickLabel','','yTickLabel','',...
                  'box','on',...
                  'titleColor','red' ...
              );
if ~( (isvector(opt.size) && numel(opt.size)==2) ||...
        (iscell(opt.size) && numel(opt.size)==numel(dat.clab)))
    error 'size does not contain the right number of elements';
end
if ~isfield(opt,'scaleMax')
    opt.scaleMax = opt.size/2;
end

%% Split and group options
fns = fieldnames(opt);
plotopt = struct();
scalpopt = struct();
for k=1:numel(fns)
    fn = fns{k};
    switch(fn)
        case 'dispChans'
            scalpopt.(fn) = opt.(fn);   % use dynamic fieldnames
        % Check if property belongs to only drawScalpOutline
        case {'showLabels' 'labelProps' 'minorLabelProps' 'markerProps' ...
                'crossSize' 'crossWidth' 'lineProps' 'markChannels' ...
                'markLabelProps' 'markMarkerProps' 'dispChans' ...
                'ears' 'reference' 'referenceProps' 'handle'}
            scalpopt.(fn) = opt.(fn);   
            opt = rmfield(opt,fn);    % Remove from opt
        % Check if property belongs to only this function
        case {'size','scalePlots','scaleMax','axis','stretch',...
                'shift'}
            % Do nothing
        otherwise       % assume it's meant for plotChannel
            plotopt.(fn) = opt.(fn);   
            opt = rmfield(opt,fn);
    end
end

%% Restrict channel set if necessary
if isfield(opt,'dispChans')
    %mnt = mnt_restrictMontage(mnt, opt.dispChans);
else
    opt.dispChans = mnt.clab;   % All channels
end
%% Draw scalp outline
clf; % Clear current figure
axes('position',[0 0 1 1])
H.scalp = drawScalpOutline(mnt,scalpopt);
axis off
%% Figure settings
set(gcf,'Units','normalized');
%% Plot graphs
for cc = 1:numel(opt.dispChans)
    chan = opt.dispChans{cc};
    chanIdx = chanind(mnt.clab,chan);
    % Check plot size
    if isvector(opt.size), sz = opt.size;
    else sz = opt.size{chanIdx}; end
    xyp = [mnt.x(chanIdx) mnt.y(chanIdx)];
    % Scale size if necessary
    switch opt.scalePlots
        case 'xlin'
            sz = (1-abs(xyp(1))) * sz + abs(xyp(1))*opt.scaleMax;
        case 'ylin'
            sz = (1-abs(xyp(2))) * sz + abs(xyp(2))*opt.scaleMax;
        case 'radial'
            dist = pdist([xyp(1) xyp(2) ; 0 0]);
            sz = (1-dist) * sz + dist*opt.scaleMax;
    end
    % Shift position
    xyp = xyp + opt.shift;
    % Stretch position if necessary
    if opt.stretch~=0
        % From Cartesian to polar coordinates
        [phi r] = cart2pol(xyp(1),xyp(2));
        % Increase distance
        r = r * opt.stretch;
        [xyp(1),xyp(2)]=pol2cart(phi,r);
    end
    %% Change electrode site from from axis coordinates to figure
    %% coordinates
    set(gcf,'CurrentAxes',H.scalp.ax);
    [xyp(1) xyp(2)] = axescoord2figurecoord(xyp(1),xyp(2));
    xyp = xyp - sz/2;   % Set site position to middle of axes
    axes('position',[xyp(1) xyp(2) sz(1) sz(2)]);
    % Plot channel
    H.chan_plot(cc) = plotChannel(dat,chan,plotopt);
    %% Check options
    if strcmp(opt.axis,'off'),axis off, end
end