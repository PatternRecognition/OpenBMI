
function h = fig2CircularSubplot(figs,varargin)
% FIG2CIRCULARSUBPLOT - arange figures in a circular subplot
%
%Description:
% Takes a number of figure handles and uses fig2subplot to arrange
% them in a circular manner.
%
%Usage:
% H = figs2CircularSubplot(HFIGS, <OPT>)
%
%Input:
% HFIGS: a vector of figure handles
% opt.
%     angles       - list of angles in degrees. 0 is centered at the top.
%                    must be at least same length as HFIGS
%     centerLegend - if 1, all legends from the indvidual figures are
%                    moved to the centre of the circular arangement.
%                    this makes sense if all figures have the same legend.
%     figWidth     - relative size of sub-figures
%     deleteFigs   - if 1, original figures are deleted.
%
%Output:
% H = Handle to the new subplot figure and its children
%
% Author(s): Thomas Rost Oct 2010

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
    'angles',[30,90,150,210,270,330],...
    'centerLegend',1,...
    'figWidth',0.25,...
    'deleteFigs',1);


fig_opt = struct();

fig_opt.deleteFigs = opt.deleteFigs;

fwidth = opt.figWidth;


% convert the angle argument to relative x-y positions
radians = opt.angles /180.  * pi;
positions = [];
c = 0.47 - 0.5*fwidth;
for i = 1:length(figs),
    p = [ 0.5 + c* sin(radians(i)), 0.5 + c* cos(radians(i)),fwidth,fwidth];
    positions = [positions;p];
end
positions(:,2) = positions(:,2)-0.5*fwidth;
positions(:,1) = positions(:,1)-0.5*fwidth;

fig_opt.positions = positions;

% create the new figure
h = fig2subplot(figs,fig_opt);


if opt.centerLegend,
    % find all legends and move them to the centre of the new figure
    legs = findall(h.main,'tag','legend');
    for i=1:length(legs),
        leg = legs(i);
        pos = get(leg,'position');
        pos(1) = 0.5 - 0.5 * pos(3);
        pos(2) = 0.5 - 0.5 * pos(4);
        set(leg,'position',pos);
        
    end
end

