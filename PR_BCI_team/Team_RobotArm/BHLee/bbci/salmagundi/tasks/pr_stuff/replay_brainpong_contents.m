function replay_brainpong_contents(number, varargin);
%function replay_brainpong_contents(number)
%
% Guido 04/03/04

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
		  'start', 0, ...
		  'stop', inf);

fs = 25;
load_log('brainpong_master',number);

tt_old = opt.start*1000;
stop = opt.stop*1000;
out = load_log;

BALL = 5;
WINNER = 77;
old_pos1 = 0;
old_pos2 = 0;
START = 10;
STOP = 11;
PAUSE = 12;
EXIT = 13;
SCORE_ON = 7;
BALL_ON = 20;
COUNTER_ON = 50;
WINNER = 77;

run = 1;

while run & ~isempty(out)
  if iscell(out) & length(out)==10
    var1 = out{6};
    var2 = out{10};
    tt  = out{8}*1000;
    if tt>stop
      run = false;
      break;
    end
    if tt<tt_old,
      continue;
    end
    
    while tt>tt_old+1000/fs
%      fprintf('%g,%g\n',tt,tt_old);
      tt_old = tt_old+1000/fs;
    end
    
    array = [var1(1),var2];
    a = [0,array,zeros(1,5-length(array))];

    switch a(2)
      
     case START 
      fprintf('%5.1f: game starts\n',tt/1000);
      
     case STOP
      fprintf('%5.1f: game stops\n',tt/1000);
      
     case COUNTER_ON 
      if a(3)
%	fprintf('%5.1f: countdown starts\n',tt/1000);
      else
	fprintf('%5.1f: countdown stops\n',tt/1000);
      end
      
     case WINNER
      fprintf('%5.1f: game over ***\n',tt/1000);
      
     case EXIT
      fprintf('%5.1f: exit\n',tt/1000);
      run = false;
      
    end
    
  end
  out = load_log;
end


load_log('exit');
