function [H, locations]= stimutil_fixationCross(varargin)
%STIMUTIL_FIXATIONCROSS - Initialize Fixation Cross for Cue Presentation
%
%H= stimutil_fixationCross(<OPT>)
%
%Arguemnts:
% OPT: struct or property/value list of optional properties:
%
%Returns:
% H - Handle to graphic objects

% blanker@cs.tu-berlin.de, Nov 2007


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'cross_vpos', 0, ...
                  'cross_size', 0.1, ...
                  'cross_spec', {'Color',0.7*[1 1 1], 'LineWidth',8}, ...
                  'loc_radius', 0.1, ...
                  'loc_distance', 0.7);

fix_w= opt.cross_size;
fix_h= fix_w;

%% if axis('equal') is *not* used.
%fp= get(gcf, 'Position');
%fix_h= fix_w/fp(4)*fp(3);

H= line([-fix_w fix_w; 0 0]', ...
        opt.cross_vpos + [0 0; -fix_h fix_h]', ...
        opt.cross_spec{:}, ...
        'Visible','off');

hold on;    

%for ii = 1:opt.speakerCount,
for ii = opt.speakerSelected,
    degrees = (ii-1)*(360/opt.speakerCount);
    x_loc = sind(degrees);
    y_loc = cosd(degrees);
    locations(ii) = circle([x_loc * opt.loc_distance, y_loc * opt.loc_distance], opt.loc_radius, 20, [0.7 0.7 0.7]);
end