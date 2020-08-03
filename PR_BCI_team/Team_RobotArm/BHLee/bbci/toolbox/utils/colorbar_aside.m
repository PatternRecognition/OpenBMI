function h_cb= colorbar_aside(varargin)
%COLORBAR_ASIDE - Add a colorbar without shrining the axis
%
%Synopsis:
% H_CB= colorbar_aside(<OPT>)
% H_CB= colorbar_aside(ORIENTATION, <OPT>)
%
%Arguments:
% ORIENTATION: see OPT.orientation
% OPT: struct or property/value list of optional properties
%  .orientation: orientation resp. location of the colorbar relative
%     to the axis: 'vert','horiz', 'NorthOutside', 'EastOutside',
%     'SouthOutside', 'WestOutside'.
%
%Returns:
% HCB: handle of the colorbar
%
%Note:
% So far, colorbar_aside is compatible with Matlab 6, but it allows
% the new orientation modes of Matlab 7.

% blanker@cs.tu-berlin.de

if mod(nargin, 2)==1,
  opt= struct('orientation', varargin{1});
  varargin(1)= [];
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'orientation', 'vert', ...
                  'gap', 0.02);

ax= gca;
pos= get(ax, 'position');
%% this has an influence on the behavious when resizing the figure:
epos= axis_getEffectivePosition(ax);
%% this would require Matlab 7:
%% h_cb= colorbar(opt.orientation);
if ismember(lower(opt.orientation), {'horiz','northoutside','southoutside'}),
  h_cb= colorbarv6('horiz');
%  h_cb= colorbar('SouthOutside');
else
  h_cb= colorbar('EastOutside');
end
set(ax, 'position',pos);
drawnow;
cb_pos= get(h_cb, 'position');
%ii= strmatch(opt.orientation, {'vert','horiz'}, 'exact');
%cb_pos(ii)= pos(ii)+pos(ii+2)+0.02;
%cb_pos(5-ii)= min(cb_pos(5-ii), epos(5-ii));
%cb_pos(3-ii)= pos(3-ii) + (pos(5-ii)-cb_pos(5-ii))/2;
switch(lower(opt.orientation)),
 case {'vert','eastoutside'},
  cb_pos(1)= pos(1) + pos(3) + opt.gap;
  cb_pos(4)= min(cb_pos(4), epos(4));
  cb_pos(2)= pos(2) + (pos(4)-cb_pos(4))/2;
 case 'westoutside',
  cb_pos(1)= pos(1) - cb_pos(3) - opt.gap;
  cb_pos(4)= min(cb_pos(4), epos(4));
  cb_pos(2)= pos(2) + (pos(4)-cb_pos(4))/2;
  set(h_cb, 'YAxisLocation','left');
 case {'horiz','southoutside'},
  cb_pos(2)= pos(2) - cb_pos(4) - opt.gap;
  cb_pos(3)= min(cb_pos(3), epos(3));
  cb_pos(1)= pos(1) + (pos(3)-cb_pos(3))/2;
 case 'northoutside',
  cb_pos(2)= pos(2) + pos(4) + opt.gap;
  cb_pos(3)= min(cb_pos(3), epos(3));
  cb_pos(1)= pos(1) + (pos(3)-cb_pos(3))/2;
  set(h_cb, 'XAxisLocation','top');
 otherwise,
  error('unknown orientation/location');
end
set(h_cb, 'Position',cb_pos);
%delete(h_cb);
%cb_ax= axes('position',cb_pos);
%h_cb= colorbar(cb_ax, 'peer',ax);

if nargout==0,
  clear h_cb;
end
