function [copy_handle, spelled_handle]= stimutil_updateCopyText(written, varargin)
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
                  'text_hpos', -1.5, ...
                  'text_size', 28, ...
                  'text_blur', [.5 .5 .5], ...
                  'text_col', [1 1 1], ...
                  'text_highlight', 44, ...
                  'text_col_highlight', [1 0 0], ...
                  'space_char', '_', ...
                  'mask_errors', 0, ...
                  'mark_errors', 1, ...
                  'fake_feedback', 0, ...
                  'no_feedback', 0);

opt.writ_text_col = opt.text_col;

if opt.auto_correct && ~sum([opt.mask_errors, opt.fake_feedback, opt.no_feedback]),
    opt.mask_errors = 1;
end
if sum([opt.mask_errors, opt.fake_feedback, opt.no_feedback]) > 1,
    error('Only one of mask_errors, fake_feedback and no_feedback can be true.');
end
              
if opt.fake_feedback,
    written = opt.spellString(1:length(written));
    opt.mark_errors = 0;
elseif opt.no_feedback,
    opt.writ_text_col = get(gcf, 'color');
    opt.mark_errors = 0;
elseif opt.mask_errors,
    oldWritten = written;
    written = opt.spellString(1:length(written));
    written(find(opt.procVar.error_list == 3)) = '.';
    opt.mark_errors = 0;
end
    
copyString = regexprep(upper(opt.spellString), ' ', opt.space_char);
writtenString = regexprep(upper(written), ' ', opt.space_char);

% DO THE COPY TEXT
char_error = 0;
lastIdx =1;

%reset all
set(opt.procVar.copyText, 'color', opt.text_blur, 'FontSize', opt.text_size);
set(opt.procVar.spelledText, 'color', opt.writ_text_col, 'FontSize', opt.text_size, 'string', '');

%do proper changes
if isempty(writtenString),
  set(opt.procVar.copyText(1), 'color', opt.text_col_highlight, 'FontSize', opt.text_highlight);
else
    if sum([opt.mask_errors, opt.fake_feedback, opt.no_feedback]) == 0, % actual feedback
      char_error = 0;
      for i=1:length(writtenString),
        if length(copyString) < i,
          set(opt.procVar.spelledText(i), 'string', writtenString(i));      
        elseif writtenString(i) == copyString(i) && ~char_error,
          set(opt.procVar.copyText(i), 'color', opt.text_col);
          set(opt.procVar.spelledText(i), 'string', writtenString(i));
        else
          if ~char_error,
            char_error = 1;
            set(opt.procVar.copyText(i), 'color', opt.text_col_highlight, 'FontSize', opt.text_highlight);
          end
          if opt.mark_errors && writtenString(i) ~= copyString(i),
            set(opt.procVar.spelledText(i), 'string', writtenString(i), 'color', opt.text_col_highlight);
          else
            set(opt.procVar.spelledText(i), 'string', writtenString(i));
          end
       end
      end
      if ~char_error && length(opt.procVar.copyText) > i,
        set(opt.procVar.copyText(i+1), 'color', opt.text_col_highlight, 'FontSize', opt.text_highlight);
      end
    else % for the fake feedbacks
        for i = 1:length(writtenString),
            if opt.procVar.error_list(i) && opt.mask_errors,
                set(opt.procVar.spelledText(i), 'color', opt.text_blur, 'string', writtenString(i));
            else
                set(opt.procVar.spelledText(i), 'color', opt.writ_text_col, 'string', writtenString(i));
            end
        end
        set(opt.procVar.copyText, 'color', opt.text_col);
        if length(written) < length(copyString),
            set(opt.procVar.copyText(length(written)+1), 'color', opt.text_col_highlight, 'FontSize', opt.text_highlight);
        end
    end
end

drawnow;

copy_handle = opt.procVar.copyText;
spelled_handle = opt.procVar.spelledText;
