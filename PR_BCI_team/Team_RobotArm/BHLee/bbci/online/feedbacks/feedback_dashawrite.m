function opt = feedback_dashawrite(fig, opt, ctrl);
%FEEDBACK_DASHAWRITE - BBCI Typewriter Feedback based on Dasher
%
%Synopsis:
% OPT = feedback_dashawrite(FIG, OPT, CTRL)
%
%Arguments:
% FIG  - handle of figure
% OPT  - struct of optional properties, see below
% CTRL - control signal to be received from the BBCI classifier
%
%Output:
% OPT - updated structure of properties
%
%Optional Properties:
% countdown: length of countdown before application starts [ms]
% background: background color
%
%Markers written to the parallel port
% 30-49: dash state changed
%  30: countdown starts
% 61-96: code of selected letter
% 200: init of the feedback
% 210: game status changed to 'play'
% 211: game status changed to 'pause'
% 212: game status changed to 'stop'
%
%See:
% tester_dashawrite, feedback_dashawrite_init

% Author(s): Benjamin Blankertz, Feb-2006

global DATA_DIR
persistent H HH state dash timer lm

if ~isfield(opt,'reset'),
  opt.reset = 1;
end

if opt.reset,
  opt.reset= 0;
  opt= set_defaults(opt, ...
		    'charset', '<ABCDEFGHIJKLMNOPQRSTUVWXYZ._',...
        'rate_control', 1, ...
        'speed', 2, ...
        'speed_forward', 1/12, ...
                    'maxangle', 20, ...
		    'countdown', 1000, ...
                    'duration_show_selected', 1000, ...
        'fieldwidth', 0.1, ...
                    'minfieldheight', 1/60, ...
                    'maxfieldheight', 1/12, ...
                    'maxfieldfontsize', 0.25, ...
        'fieldcolor', [1 0.7 0.1; 1 0.9 0.7], ...
                    'pointer_width', 0.1, ...
                    'pointerline_spec', {}, ...
		    'msg_spec', {'FontSize',0.2}, ...
		    'text_spec', {'FontSize',0.3, 'FontName','Courier', 'FontWeight','bold'}, ...
		    'textfield_width', 0.8, ...
        'textfield_height', 0.2, ...
        'textfield_box_spec', {'Color',[0 0.5625 0], 'LineWidth',3}, ...
		    'language_model', 'german', ...
		    'lm_headfactor', [0.85 0.85 0.75 0.5 0.25], ...
		    'lm_letterfactor', 0.01, ...
		    'lm_npred', 2, ...
		    'lm_probdelete', 0.1, ...
        'background', 0.9*[1 1 1], ...
		    'fs', 25, ...
		    'text_reset', 0, ...
        'time_after_text_reset', 3000, ...
        'log', 0, ...
		    'parPort', 1, ...
		    'status', 'pause', ...
        'changed', 1, ...
        'position', get(fig,'position'));

  if isempty(opt.language_model),
    lm= [];
  else
    lm= lm_loadLanguageModel(opt.language_model);
    lm.charset= [lm.charset, '<'];
  end
  
  [HH, dash]= feedback_dashawrite_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);

  do_set('init',handles, 'dashawrite',opt);
  do_set(200);

  dash.written= '';
  dash.written_ctrl= '';
  dash.laststate= NaN;
  dash.lastdigit= NaN;
  dash.laststatus= 'urknall';
  dash.ctrl= 0;
  dash.x= 0;
  dash.y= 0.5;
  dash.i1= NaN;
  dash.i2= NaN;
  dash= rescale_letterfields(H, HH, dash, lm, opt);

  timer.msec= 0;
  state= -1;
end

if opt.changed,
  if ~strcmp(opt.status,dash.laststatus),
    dash.laststatus= opt.status;
    switch(opt.status),
     case 'play',
      do_set(210);
      state= 0;
     case 'pause',
      do_set(211);
      do_set(H.msg, 'String','pause', 'Visible','on');
      dash.arrow_len= 0;
      state= -1;
     case 'stop';
      do_set(212);
      do_set(H.msg, 'String','stopped', 'Visible','on');
      state= -2;
    end
  end
end

if opt.text_reset,
  opt.text_reset= 0;
  dash.written= '';
  do_set(H.textfield, 'String','');
  dash.written_ctrl= [dash.written_ctrl '|'];
  opt.changed= 1;
  timer.msec= 0;
  opt.countdown= opt.time_after_text_reset;
  state= 0;
end

if opt.changed==1,
  opt.changed= 0;
  %% TODO (?): label_radfactor
  do_set(H.textfield, opt.text_spec{:});
  do_set(H.msg, opt.msg_spec{:});
  if ~isempty(opt.language_model),
    lm= lm_loadLanguageModel(opt.language_model);
    lm.charset= [lm.charset, '<'];
  end
  if strcmp(opt.status, 'play') & state~=0,
    do_set(H.msg, 'Visible','off');
  end
end

if opt.rate_control,
  dash.ctrl= dash.ctrl + ctrl*opt.speed/opt.fs;
  dash.ctrl= min(1, max(-1, dash.ctrl));
else
  dash.ctrl= ctrl;
end

if state~=dash.laststate,
  dash.laststate= state;
  if state>=0,
    do_set(30+state);
  end
  fprintf('state: %d\n', state);
end

if state==0,
  if timer.msec>opt.countdown,
    do_set(H.msg, 'Visible','off');
    state= 1;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
  digit= ceil((opt.countdown-timer.msec)/1000);
  if digit~=dash.lastdigit,
    do_set(H.msg, 'String',int2str(digit), 'Visible','on');
    dash.lastdigit= digit;
  end
end

if state==1,  %% scale letter fields according to probabilities
  dash= rescale_letterfields(H, HH, dash, lm, opt);
  state= 2;
end

if state==2,  %% choose target
  dx= opt.speed_forward/opt.fs;
  dash.x= dash.x + dx;
%  fprintf('%.5f\r', dash.x);
  angle= dash.ctrl * opt.maxangle;
  dash.y= dash.y + dx*tan(angle/180*pi);
  dash.y= min(1, max(0, dash.y));
  if dash.x >= 1,
    state= 3;
  end
end

if state==3,  %% target was selected
  [mi,dash.selected]= max(find(dash.sect(1:end-1)>=dash.y));
  do_set(60+dash.selected);
  do_set(H.letter(dash.selected), 'Color',[0 0.7 0], 'FontWeight','bold');
  written_char= opt.charset(dash.selected);
  dash.written_ctrl= [dash.written_ctrl, written_char];
  fprintf('written: %s\n', dash.written_ctrl);
  if written_char=='<',
    dash.written= dash.written(1:max(0,end-1));
  else
    dash.written= [dash.written, written_char];
  end
  writ= [dash.written '_'];
  iBreaks= find(writ=='_');
  ll= 0;
  clear textstr;
  while length(iBreaks)>0,
    ll= ll+1;
    linebreak= iBreaks(max(find(iBreaks<dash.textfield_nChars)));
    if isempty(linebreak),
      %% word too long: insert hyphenation
      linebreak= dash.textfield_nChars;
      writ= [writ(1:linebreak-1) '-' writ(linebreak:end)];
    end
    textstr{ll}= writ(1:linebreak);
    writ(1:linebreak)= [];
    iBreaks= find(writ=='_');
  end
  textstr{end}= textstr{end}(1:end-1);
  textstr= textstr(max(1,end-dash.textfield_nLines+1):end);
  do_set(H.textfield, 'String',textstr);
  timer.msec= 0;
  state= 5;
end

if state==5,  %% show selected letter
  if timer.msec>opt.duration_show_selected,
    do_set(H.letter(dash.selected), 'Color',[0 0 0], 'FontWeight','normal');
    dash.x= 0;
    dash.y= 0.5;
    state= 1;
  else
    timer.msec= timer.msec + 1000/opt.fs;
  end
end


if dash.x<1,
  xw= 1+opt.fieldwidth;
  dash.YLim= dash.y + [-0.5 0.5]/xw*(xw-dash.x);
%  if dash.YLim(1)<0,
%    dash.YLim= dash.YLim - dash.YLim(1);
%  elseif dash.YLim(2)>1,
%    dash.YLim= dash.YLim - (dash.YLim(2)-1);
%  end
end
x0= 1/(1+opt.fieldwidth);
x1= (1-dash.x)*x0;
xx= opt.fieldwidth/(1-x1)*x1;
do_set(H.dash_ax, 'XLim',[1-xx 1+opt.fieldwidth], ...
       'YLim', dash.YLim);
[mi,i1]= max(find([inf dash.sect]>dash.YLim(2)));
[mi,i2]= max(find(dash.sect>dash.YLim(1)));
%i1= max(1, i1 - 1);
%i1= max(1, i1);  %% not needed if line above is in
%i2= i2 - 1;
i2= min(length(H.letter), i2);  %% not needed if line above is in
invisible= setdiff(1:length(H.letter), i1:i2);
if ~isempty(invisible),
  do_set(H.letter(invisible), 'Visible','off');
end
% $$$ if ~isnan(dash.i1),
% $$$   pos= get(HH.letter(dash.i1), 'Position');
% $$$   pos(2)= mean(dash.sect(dash.i1+[0 1]));
% $$$   do_set(H.letter(dash.i1), 'VerticalAli','middle', ...
% $$$          'Position',pos);
% $$$   pos= get(HH.letter(dash.i2), 'Position');
% $$$   pos(2)= mean(dash.sect(dash.i2+[0 1]));
% $$$   do_set(H.letter(dash.i2), 'VerticalAli','middle', ...
% $$$          'Position',pos);
% $$$ end
for ii= i1:i2,
  fontsz= min(dash.lettersize(ii)/diff(dash.YLim), opt.maxfieldfontsize);
  do_set(H.letter(ii), 'FontSize', fontsz, 'Visible','on');
end
% $$$ dash.i1= i1;
% $$$ dash.i2= i2;
% $$$ if i2-i1<6, keyboard; end
% $$$ if i1<i2,
% $$$   pos= get(HH.letter(dash.i1), 'Position');
% $$$   pos(2)= dash.sect(dash.i1+1);
% $$$   do_set(H.letter(dash.i1), 'VerticalAli','baseline', ...
% $$$          'Position',pos);
% $$$   pos= get(HH.letter(dash.i2), 'Position');
% $$$   pos(2)= dash.sect(dash.i2);
% $$$   do_set(H.letter(dash.i2), 'VerticalAli','cap', ...
% $$$          'Position',pos);
% $$$ end
%max(dash.lettersize(i1:i2)/diff(dash.YLim))

%% for debugging purpose
%opt.dash= dash;
%opt.state= state;
%opt.timer= timer;
%opt.lm= lm;

do_set('+');
return;





function dash= rescale_letterfields(H, HH, dash, lm, opt)

prob= lm_getProbability(lm, dash.written, opt);
dashprob= zeros(length(opt.charset), 1);
for li= 1:size(dashprob,1),
  ii= find(lm.charset==opt.charset(li));
  dashprob(li)= prob(ii);
end
y0= 1;
dash.sect(1)= y0;
dashprob= max(opt.minfieldheight, dashprob);
dashprob= min(opt.maxfieldheight, dashprob);
dash.fieldheight= dashprob/sum(dashprob);
for ii= 1:length(opt.charset),
  y1= y0 - dash.fieldheight(ii);
  dash.sect(ii+1)= y1;
  do_set(H.field(ii), 'YData',[y1 y1 y0 y0]);
  dash.lettersize(ii)= dash.fieldheight(ii) * dash.extent_factor;
  pos= get(HH.letter(ii), 'Position');
  pos(2)= (y0+y1)/2;
  do_set(H.letter(ii), 'Position',pos, 'FontSize',dash.lettersize(ii));
  y0= y1;
end
dash.sect(end)= 0;
