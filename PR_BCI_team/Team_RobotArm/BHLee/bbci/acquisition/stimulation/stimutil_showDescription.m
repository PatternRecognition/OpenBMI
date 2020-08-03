function HANDLE= stimutil_showDescription(desc, varargin)
%STIMUTIL_SHOWDESCRIPTION - Show a Description and Wait
%
%Synopsis:
% HANDLE= stimutil_showDescription(DESC, <OPT>)
%
%Arguments:
% DESC: String or cell array of strings.
% OPT: struct or property/value list of optional arguments:
% 'handle_msg': Handle to text object which is used to display the countdown
%    message. If empty a new object is generated. Default [].
% 'handle_background': Handle to axis object on which the message should be
%    rendered. If empty a new object is generated. Default [].
% 'desc_textspec': Cell array. Text object specifications for description
%   text object. Default: {'FontSize',0.05, 'Color',[0 0 0]})
% 'waitfor': Stop criterium. Possible values are numeric values specifying
%   the time to wait in seconds, the string 'key' which means waiting
%   until the experimentor hits a key in the matlab console or a string
%   or cell arrow of strings which is interpreted as marker descriptions
%   for which the EEG is scanned then (e.g. 'R*' waits until some response
%   marker is acquired).
%   Use value 0 for no waiting.
% 'delete': Delete graphic objects at the end. Default 1, except for
%   opt.waitfor=0.
%
%Returns:
% HANDLE: Struct of handles to graphical objects (only available for
%    opt.delete=0).

% blanker@cs.tu-berlin.de, Jul-2007

global VP_SCREEN

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'clf', 0, ...
                  'handle_background', [], ...
                  'desc_maxsize', [0.9 0.8], ...
                  'desc_textspec', {'FontSize',0.05, 'Color',.0*[1 1 1]}, ...
                  'desc_pos', [0.5 0.5], ...
                  'desc_boxgap', 0.05, ...
                  'delete', 1, ...
                  'position', VP_SCREEN, ...
                  'waitfor', 'R*', ...
                  'waitfor_msg', 'Press <ENTER> to continue: ');

if isequal(opt.waitfor, 0),
  opt.delete= 0;
end

HANDLE= [];
if opt.clf,
  clf;
  set(gcf, 'Position',opt.position);
  set(gcf, 'ToolBar','none', 'MenuBar','none');
end

if isempty(get(gcf, 'Children')),
  memo.axis= [];
else
  memo.axis= gca;
end
h.axis= axes('Position',[0 0 1 1]);
set(h.axis, 'XLim',[0 1], 'YLim',[0 1], 'Visible','off');

if iscell(desc),
  %% Description has linebreaks:
  %% Choose size of the description box by trial and error
  col_background= get(gcf, 'Color');
  factor= 1;
  too_small= 1;
  nLines= length(desc);
  nChars= max(apply_cellwise2(desc, 'length'));
  while too_small,
    desc_fontsize= factor * min( opt.desc_maxsize./[nChars nLines] );
    ht= text(opt.desc_pos(1), opt.desc_pos(2), desc);
    set(ht, 'FontUnits','normalized', 'FontSize',desc_fontsize, ...
            'Color',col_background, 'HorizontalAli','center');
    drawnow;
    rect= get(ht, 'Extent');
    too_small= rect(3)<opt.desc_maxsize(1) & rect(4)<opt.desc_maxsize(2);
    if too_small,
      factor= factor*1.1;
    end
    delete(ht);
  end
  factor= factor/1.1;
  %% render description text
  desc_fontsize= factor * min( opt.desc_maxsize./[nChars nLines] );
  h.text= text(opt.desc_pos(1), opt.desc_pos(2), desc);
  set(h.text, opt.desc_textspec{:}, 'HorizontalAli','center', ...
              'FontUnits','normalized', 'FontSize',desc_fontsize);
else
  %% Description is given as plain string:
  %% Determine number of characters per row
  textfield_width= opt.desc_maxsize(1);
  textfield_height= opt.desc_maxsize(2);
  ht= text(0, 0, {'MMMMMMM','MMMMMMM','MMMMMMM','MMMMMMM','MMMMMMM'});
  set(ht, 'FontName','Courier New', 'FontUnits','normalized', ...
          opt.desc_textspec{:});
  rect= get(ht, 'Extent');
  char_width= rect(3)/7;
  linespacing= rect(4)/5;
  char_height= linespacing*0.85;
  textfield_nLines= floor((textfield_height-2*char_height)/linespacing)+2;
  textfield_nChars= floor(textfield_width/char_width);
  delete(ht);
  h.text= text(opt.desc_pos(1), opt.desc_pos(2), {' '});
  set(h.text, 'HorizontalAli','center', 'FontUnits','normalized', ...
              opt.desc_textspec{:});

  %% Determine linebreaking.
  writ= [desc ' '];
  iBreaks= find(writ==' ');
  ll= 0;
  clear textstr;
  while length(iBreaks)>0,
    ll= ll+1;
    linebreak= iBreaks(max(find(iBreaks<textfield_nChars)));
    if isempty(linebreak),
      %% word too long: insert hyphenation
      linebreak= textfield_nChars;
      writ= [writ(1:linebreak-1) '-' writ(linebreak:end)];
    end
    textstr{ll}= writ(1:linebreak);
    writ(1:linebreak)= [];
    iBreaks= find(writ==' ');
  end
  textstr{end}= textstr{end}(1:end-1);
  textstr= textstr(max(1,end-textfield_nLines+1):end);
  set(h.text, 'String',textstr);
end

drawnow;
rect= get(h.text, 'Extent');
set(h.text, 'Position',[rect(1) opt.desc_pos(2), 0]);
set(h.text, 'HorizontalAli','left');

%% render description frame
h.frame= line([rect(1)+rect(3) rect(1)+rect(3) rect(1) rect(1); ...
               rect(1)+rect(3) rect(1) rect(1) rect(1)+rect(3)] + ...
              opt.desc_boxgap*[1 1 -1 -1; 1 -1 -1 1], [rect(2)+rect(4) ...
                    rect(2) rect(2) rect(2)+rect(4); rect(2) rect(2) ...
                    rect(2)+rect(4) rect(2)+rect(4)] + opt.desc_boxgap*[1 ...
                    -1 -1 1; -1 -1 1 1]);
set(h.frame, 'LineWidth',2, 'Color',[0 0 0]);

if ~isempty(opt.waitfor),
  if isnumeric(opt.waitfor),
    pause(opt.waitfor),
  elseif isequal(opt.waitfor, 'key'),
    fprintf(opt.waitfor_msg);
    pause;
    fprintf('\n');
  else
    stimutil_waitForMarker(opt, 'stopmarkers',opt.waitfor);
  end
end

if isequal(opt.delete, 'fig'),
  close;
elseif opt.delete,
  delete(h.axis);
  drawnow;
else
  if ~isempty(memo.axis),
    axes(memo.axis);
  end
  if nargout>0,
    HANDLE= h;
  end
end
