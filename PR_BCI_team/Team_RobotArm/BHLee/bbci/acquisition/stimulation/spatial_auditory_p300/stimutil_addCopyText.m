function [copy_handle, spelled_handle]= stimutil_addCopyText(varargin)
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
                  'text_vpos', .9, ...
                  'text_hpos', 'center', ...
                  'text_size', 28, ...
                  'text_blur', [.5 .5 .5], ...
                  'text_col', [1 1 1], ...
                  'text_highlight', 44, ...
                  'text_col_highlight', [1 0 0], ...
                  'space_char', '_');

if ~isempty(opt.spellString),
  copyString = regexprep(upper(opt.spellString), ' ', opt.space_char);
end
set(gca,'FontName','FixedWidth');

x_ax = get(gca, 'XLim');
y_ax = get(gca, 'YLim');
axis_dim = get(gcf, 'Position');
x_fi = axis_dim(3);y_fi = axis_dim(4);

x_pi = (x_ax(2)-x_ax(1))/x_fi;
y_pi = (y_ax(2)-y_ax(1))/y_fi;

if ischar(opt.text_hpos) && strcmp(opt.text_hpos, 'center'),
    opt.text_hpos = -(length(copyString)*x_pi*1.8*opt.text_size)/2;
end
if opt.visualize_text && ~isempty(opt.spellString),
  for i = 1:length(copyString),
    if i == 1,
      colS = opt.text_col_highlight;
      textS = opt.text_highlight;
    else
      colS = opt.text_blur;
      textS = opt.text_size;
    end
    copy_handle(i) = text(opt.text_hpos+((i-1)*x_pi*1.8*opt.text_size), opt.text_vpos, copyString(i), 'color', colS, 'fontsize', textS, 'HorizontalAlignment', 'center','interpreter','none');
  end
else
  copy_handle = [];
end

for i = 1:60,
  spelled_handle(i) = text(opt.text_hpos+((i-1)*x_pi*1.8*opt.text_size), opt.text_vpos-y_pi*2*opt.text_size, ' ', 'color', [1 1 1], 'fontsize', opt.text_size, 'HorizontalAlignment', 'center','interpreter','none');
end