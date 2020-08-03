function varargout= stimutil_fixationCrossandLocations(varargin)
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
                  'loc_distance', 0.7, ...
                  'angle_offset', 0, ...
                  'label_holders', 0, ...
                  'tick_marks', 0, ...
                  'tick_size', .5);

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

label_holders = [];
tick_marks = [];

if opt.label_holders,
  label_holders = zeros(length(opt.speakerSelected),length(opt.speakerSelected)+1);
end
if opt.tick_marks,
  tick_marks{1} = fill([0 .8 0 -.5 0]*opt.tick_size, [.2 .8 -.2 .5 .2]*opt.tick_size, [.3 .8 0]);
  tick_marks{2}(1) = line([-.5 .5]*opt.tick_size, [-.5 .5]*opt.tick_size, 'color', [1 0 0], 'LineWidth', 30);
  tick_marks{2}(2) = line([.5 -.5]*opt.tick_size, [-.5 .5]*opt.tick_size, 'color', [1 0 0], 'LineWidth', 30);
end
%for ii = 1:opt.speakerCount,
for ii = opt.speakerSelected,
    degrees = opt.angle_offset + (ii-1)*(360/length(opt.speakerSelected));
    x_loc = sind(degrees);
    y_loc = cosd(degrees);
    x_cent = x_loc * opt.loc_distance;
    y_cent = (y_loc * opt.loc_distance)+opt.cross_vpos;
    locations(ii) = circle([x_cent, y_cent], opt.loc_radius, 20, [0.8 0.8 0.8]);
    if opt.label_holders,
      label_holders(ii,1) = text(x_cent, y_cent, '', 'color', [0 0 0], 'fontsize', 30, 'HorizontalAlignment', 'center','interpreter','none');
      for jj = opt.speakerSelected,
          degrees = opt.angle_offset + (jj-1)*(360/length(opt.speakerSelected));
          x_loc = sind(degrees);
          y_loc = cosd(degrees);
          label_holders(ii,jj+1) = text(x_cent+x_loc*.7*opt.loc_radius, y_cent+y_loc*.7*opt.loc_radius, '', 'color', [0 0 0], 'fontsize', 30, 'HorizontalAlignment', 'center','interpreter','none');
      end
    end
end

varargout(1) = {H};
varargout(2) = {locations};
if nargout >= 3,
  varargout(3) = {label_holders};
end
if nargout >= 4,
    varargout(4) = {tick_marks};
end