function stim= stim_checkerboard(state, varargin)
%stim= stim_checkerboard(state, <opt>)

%     opt
%             .minEventsLeft - break only if at least this many events left

%global IO_ADDR 

if ~isstruct(state),
  error('first argument must be a struct');
end
if ~isfield(state, 'param'),
  [state.param]= deal({});
end
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
		  'bkgColor',0.3*[1 1 1], ...
		  'textColor',0*[1 1 1], 'textPos',0.55, 'textSize',0.1, ...
		  'crossColor',0.7*[1 1 1], 'crossLineWidth',4, ...
		  'stimPos',0.55, 'stimSize',0.25, 'stimColor',[0 0 0], ...
		  'maxExtent',1, ...
		  'trialTiming',[3 1 0.5], ...
		  'breakFreq',inf, 'minEventsLeft',7, ...
		  'relaxTimeBetween',[], 'breakFactor',1, ...
		  'msgRelax','entspannen', 'msgBreak','kurze Pause', ...
		  'msgContinue','Taste drücken um fortzufahren',...
  		  'squares', 7, ...
		  'board_width', 0.17, ...
		  'blink_interval', 500, ...
		  'presentation_time', 3000, ...
		  'colors', [1 1 1; 0 0 0]);

%opt.stimColor= [0.6 0.1 0.1];  %% for Klaus
if length(opt.breakFreq)==1,
    opt.breakFreq= [opt.breakFreq 10];
end
if length(opt.relaxTimeBetween)==1,
  opt.relaxTimeBetween= [opt.relaxTimeBetween, opt.trialTiming(2:3)];
end
if length(opt.trialTiming)==3
    opt.trialTiming(4)=0;
end 
%opt.msgBreak= 'short break';
%opt.msgRelax= 'relax';

nClasses= length(state);
nEvents= sum([state(:).nEvents]);

events= [];
for cc= 1:nClasses,
    events= cat(1, events, cc*ones(state(cc).nEvents,1));
end
events= events(randperm(nEvents));
nBreaks= floor((nEvents-opt.minEventsLeft)/opt.breakFreq(1));
fprintf('To stop during ''%s'' press space-bar in activated figure.\n', ...
	opt.msgBreak);
tt= opt.trialTiming(1:3);
if ~isempty(opt.relaxTimeBetween),
  tt= tt+opt.relaxTimeBetween;
end
fprintf('estimated experiment time: %.2f min [+ %.2f min pause]\n', ...
        nEvents*tt*[1;1;0.5]/60, ...
        (2+58*opt.breakFactor)/60 + nBreaks*(opt.breakFreq(2)+5)/60);

oldPos= figureMaximize;
clf;
set(gcf, 'color',opt.bkgColor);
set(gcf,'KeyPressFcn', ...
	'global pausi;pausi = [pausi,get(gcf,''CurrentCharacter'')];');
set(gcf,'DoubleBuffer','on');
set(gcf, 'units','pixel');
fig_pos= get(gcf, 'position');
width= ceil(fig_pos(3)*opt.board_width);
height= width;

h_mainax= axes;
set(h_mainax, 'unit','normalized', ...
	      'position', [0 0 1 1]);
ypos= round((fig_pos(4)-height)/2);
ax_pos= [1 ypos width height];
h_imageax= axes;
set(h_imageax, 'unit','pixel', ...
	'position', ax_pos);

im_board= toeplitz(mod(1:opt.squares,2));
colormap(opt.colors);

h_im= image(im_board+1);
set(h_imageax, 'Visible','off');
set(h_im, 'Visible','off');

axes(h_mainax);
if opt.breakFactor>=0,
  ppTrigger(251);
  h_text= showText(opt.msgRelax, opt.textSize, opt.textPos, opt.textColor);
  set(h_text, 'color', opt.textColor); 
  pause(1+39*opt.breakFactor);
  set(h_text, 'string',' ');
  h_cross= line([.48 .52; .5 .5]', 0.05+[.5 .5; .475 .525]', ...
                'color',opt.crossColor, 'lineWidth',opt.crossLineWidth); 
  axis([0 1 0 1]);
  pause(1);
  anounce_start(h_text);
  ppTrigger(252);
  pause(1);
end

h_stim= showText(' ', opt.stimSize, opt.stimPos, opt.textColor);
set(h_stim, 'color', opt.stimColor); 

% move h_stim to the background (behind the cross)
hc= get(gca, 'children');
set(gca, 'children',hc([2:end 1]));

stim= cell(1, nEvents);
for ei= 1:nEvents,
  cc= events(ei);
  if length(state(cc).stimulus)==1,
    stim{ei}= state(cc).stimulus;
  else
    stim{ei}= feval(state(cc).stimulus, state(cc).param{:});
  end
  set(h_stim, 'string',' ');

  if stim{ei}==1,
    ax_pos(1)= 1;
  else
    ax_pos(1)= fig_pos(3) - width;
  end
  set(h_imageax, 'Position',ax_pos);

  presentation_time= opt.trialTiming(1)+opt.trialTiming(4)*rand;
  set(h_im, 'Visible','on');
  ppTrigger(events(ei));
  t0= clock;
  while etime(clock, t0)<presentation_time,
    pause(opt.blink_interval/1000);
    set(gcf, 'Colormap',flipud(get(gcf, 'Colormap')));
    ppTrigger(101);
    drawnow;
  end
  set(h_im, 'Visible','off');
  ppTrigger(100);
  pause(opt.trialTiming(2)+opt.trialTiming(3)*rand);
  if ~isempty(opt.relaxTimeBetween),
    set(h_stim, 'string', 'x', 'fontSize',opt.stimSize);
    ppTrigger(nClasses+1);
    pause(opt.relaxTimeBetween(1));
    set(h_stim, 'string', ' ');
    ppTrigger(100);
    pause(opt.relaxTimeBetween(2)+opt.relaxTimeBetween(3)*rand);
  end
   
  if mod(ei, opt.breakFreq(1))==0 & ei<nEvents-opt.minEventsLeft & ...
    opt.breakFactor>=0,
    give_him_a_break(opt.breakFreq(2), h_text, opt);
  end
end

delete([h_stim; h_cross]);
set(h_text, 'string',opt.msgRelax);
ppTrigger(253);
pause(1+9*opt.breakFactor);

ppTrigger(254);
clf;
figureRestore(oldPos);




function give_him_a_break(breakLength, h_text, opt)

global pausi
ppTrigger(249);
set(h_text, 'string',opt.msgBreak);
pausi = '';
pause(breakLength);
if strfind(pausi,' ');
    set(h_text,'string',opt.msgContinue);
    pause; 
end
set(h_text, 'string',' ');
pause(1);
anounce_start(h_text);
ppTrigger(250);
pause(1);
