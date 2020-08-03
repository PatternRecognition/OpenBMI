function H= stimutil_fixationCrosses(varargin)
%STIMUTIL_FIXATIONCROSSES - Initialize Fixation Cross for Cue Presentation
%
%H= stimutil_fixationCrosses(<OPT>)
%
%Arguemnts:
% OPT: struct or property/value list of optional properties:
%
%Returns:
% H - Handle to graphic objects

% blanker@cs.tu-berlin.de, Nov 2007


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'cross_size', 0.1, ...
                  'cross_spec', {'MarkerSize',40, 'LineWidth',4});

hold on;
H= plot(opt.cross_size*[1 0 0 0 -1]+[-1 0 0 0 1], ...
        opt.cross_size*[0 1 0 -1 0]+[0 -1 0 1 0], 'k+', ...
        opt.cross_spec{:});
hold off;
axis off;
