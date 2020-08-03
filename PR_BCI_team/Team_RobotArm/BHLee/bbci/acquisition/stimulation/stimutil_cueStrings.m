function H= stimutil_cueStrings(strs, varargin)
%STIMUTIL_CUESTRINGS - Initialize Strings for Cue Presentation
%
%H= stimutil_cueStrings(STRS, <OPT>)
%
%Arguemnts:
% STRS: Cell array of strings.
% OPT: struct or property/value list of optional properties:
%  'cue_hpos':
%  'cue_vpos':
%  'cue_spec': Cell array of text properties
%
%Returns:
% H - Handle to text objects

% blanker@cs.tu-berlin.de, Nov 2007

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'cue_hpos', 0, ...
                  'cue_vpos', 0.1, ...
                  'cue_spec', {'FontSize',0.2, 'Color',[0 0 0]});

cue_spec= {'HorizontalAli','center', ...
           'VerticalAli','middle', ...
           'FontUnits','normalized', ...
           'Visible','off', ...
           opt.cue_spec{:}};

nStrs= length(strs);
H= cell(1, nStrs);
for cc= 1:nStrs,
  H{cc}= text(opt.cue_hpos, opt.cue_vpos, strs{cc}, cue_spec{:});
end
