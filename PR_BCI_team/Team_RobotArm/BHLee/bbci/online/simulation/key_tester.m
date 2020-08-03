function key_tester(setup_name,varargin)
%KEY_TESTER - Test BBCI Feedback Applications with Keyboard Control
%
%Use <ESC> to quit.
%
%Synopsis:
%  key_tester(SETUP_NAME)
%
%Arguments:
%  SETUP_NAME: name of a key_tester setup file (without the prefix
%     'key_test_setup_'. That script must define the variable 'fb_opt'.
%     You can specify the keys for control by defining fb_opt.ctrl. By
%     default fb_opt.ctrl= {[28 29]} is used, meaning that one variable
%     is controlled (length of cell array) and that ASCII 28 (left arrow)
%     sets the control variable to -1, ASCII 29 (right arrow) sets the
%     control variable to 1 and all other keys set it to 0. The variable
%     fb_opt is passed to the feedback function.
%
%Examples:
%  key_tester('hexospell');


global char_wait

eval(['key_tester_setup_' setup_name]);

for i = 1:2:length(varargin)
  eval(sprintf('fb_opt.%s = varargin{%d};',varargin{i},i+1));
end


if ~isfield(fb_opt,'key_time')
  fb_opt.key_time = 250;
end

if ~isfield(fb_opt,'ctrl'),
  fb_opt.ctrl= {[28,29]};
end  

if ~isfield(fb_opt,'relevant_keys')
  fb_opt.relevant_keys = cat(2,fb_opt.ctrl{:},27);
end
  
ctrl = cell(1,length(fb_opt.ctrl));
for i = 1:length(fb_opt.ctrl)
  ctrl{i} = 0;
end
noise = [];
for i = 1:length(fb_opt.ctrl)
  noise = cat(1,noise,interp(randn(1,2000), 8));
end

noisepos = 1;

fb_opt.reset = 1;
fb_opt.status = 'play';
fb_opt.changed = 1;
fig = figure;
set(fig,'KeyPressFcn','global char_wait; char_wait = [char_wait,double(get(gcbo,''CurrentCharacter''))];set(gcbo,''CurrentCharacter'','' '');');
waitForSync;
tic;
lastkey = -inf;
newkey = false;

goodbye = false;
while ~goodbye
  fb_opt = feval(fb_opt.type,fig,fb_opt,ctrl{:});
  if toc-lastkey>fb_opt.key_time/1000
    if newkey 
      newkey = false;
      for i = 1:length(fb_opt.ctrl)
        ctrl{i} = 0;
      end
    end
    for i = 1:length(fb_opt.ctrl)
      ctrl{i}= max(-1, min(1, ctrl{i} + noise(i,noisepos)/10));
    end
    noisepos= noisepos +1;
  end
  waitForSync(40);
  char_wait = intersect(char_wait,fb_opt.relevant_keys);
  while ~isempty(char_wait)
    cc = char_wait(1);
    char_wait = char_wait(2:end);
    if cc==27
      goodbye = true;
      break;
    end
    for i = 1:length(fb_opt.ctrl)
      ccc = find(double(fb_opt.ctrl{i})==cc);
      if isempty(ccc)
          ctrl{i} = 0;
      elseif ccc==1
        ctrl{i} = -1;
        lastkey = toc;
        newkey = true;
      elseif ccc==2
        ctrl{i} = 1;
        lastkey = toc;
        newkey = true;
      else
        error;
      end
    end
  end
end

close(fig);
