function audio= replay_brainpong_new(number, varargin);
%replay_brainpong_new(NUMBER, <OPT>)
%
%Arguments
%  NUMBER - number of the generated logfile.
%  OPT - struct or property/value list of optional properties:
%   .start: time to start replay [s]
%   .stop: time to stop replay [s]
%   .save: file name to save replay, [] for not saving (default)
%   .speedup: number between 0 and 1: 0=real time, 1:fast scan
%   .max_length: split movie files, each having maximally this length [s]
%
% Guido 04/03/04, Benjamin 01-2006

global DATA_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
		  'start', 0, ...
		  'stop', inf, ...
		  'save', [], ...
		  'position',ceil(get(0,'ScreenSize')*0.5), ...
		  'speedup',0, ...
		  'centerline',1, ...
		  'blackfield',1, ...
		  'ballissquare',1, ...
		  'batheight',0.125, ...
		  'freeze_end',0, ...
		  'sound',1, ...
      'compression','none', ...
      'quality',100, ...
      'max_length',inf, ...
		  'bat1_sound', [], ...
		  'bat2_sound', [], ...
		  'border_sound', [], ...
		  'ballout_sound', [], ...
		  'bat1_audio', [], ...
		  'bat2_audio', [], ...
		  'border_audio', [], ...
		  'ballout_audio', []');

%		  'bat1_sound', [DATA_DIR 'audio/A5.wav'], ...
%		  'bat2_sound', [DATA_DIR 'audio/G5.wav'], ...
%		  'border_sound', [DATA_DIR 'audio/E5.wav'], ...

if ischar(opt.bat1_sound),
  [opt.bat1_audio,audio.fs]= wavread(opt.bat1_sound);
end
if ischar(opt.bat2_sound),
  [opt.bat2_audio,audio.fs]= wavread(opt.bat2_sound);
end
if ischar(opt.border_sound),
  [opt.border_audio,audio.fs]= wavread(opt.border_sound);
end
if ischar(opt.ballout_sound),
  [opt.ballout_audio,audio.fs]= wavread(opt.ballout_sound);
end
ae= 0;

if opt.blackfield,
  opt.fieldcolor= [0.3 0.3 0.3];
  opt.bat_color= [1 1 1]*0.9;
  opt.ball_color= [1 1 1]*0.9;
  opt.ballout_color= [1 0 0];
  opt.text_color= [1 1 1]*0.9;
  opt.centerline_color= [1 1 1]*0.9;
else
  opt.fieldcolor= 'w';
  opt.bat_color= [0 0 0];
  opt.ball_color= 'k';
  opt.ballout_color= [1 0 0];
  opt.text_color= 'k';
  opt.centerline_color= 'k';
end

fs = 25;
load_log('brainpong_master',number);

waitForSync(0);
tt_start = opt.start*1000;
tt_old = opt.start*1000;
stop = opt.stop*1000;
out = load_log;
BALL = 5;
WINNER = 77;
old_pos1 = 0;
old_pos2 = 0;
ori = 1;
moveh= 0;
movev= 0;
audio_offset= tt_start - 1000/fs - 25;

START = 10;
STOP = 11;
PAUSE = 12;
EXIT = 13;
PLAYER1 = 1;
PLAYER2 = 2;
BALL = 5;
SCORE_ON = 7;
SCORE1 = 8;
SCORE2 = 9;
BALL_ON = 20;
DIAM = 21;
BALL_COLOR = 22;
BACKGROUND_COLOR = 27;
BAT_COLOR1 = 31;
BAT_COLOR2 = 41;
BAT_WIDTH1 = 32;
BAT_WIDTH2 = 42;
BAT_HEIGHT1 = 33;
BAT_HEIGHT2 = 43;
COUNTER_ON = 50;
COUNTER = 51;
POSITION = 70;
VIEW = 60;
WINNER = 77;

run = 1;

batwidth = 1;
batheight = 0.1;

batwidth1 = batwidth;
batwidth2 = batwidth;
batheight1 = opt.batheight;
batheight2 = opt.batheight;

clf;
set(gcf,'Position', opt.position);
set(gcf,'Menubar','none');
set(gcf,'DoubleBuffer','on');
set(gcf,'NumberTitle','off');
%set(gcf,'Units','normalized');
set(gcf,'Color',opt.fieldcolor);
set(gca,'XLim',[-1 1]);
set(gca,'YLim',[-1 1]);
set(gca,'Position',[0 0 1 1]);
if opt.centerline,
  line([-1 1], [0 0], 'LineWidth',4, 'LineStyle','--', ...
       'Color',opt.centerline_color); 
end
axis off;
bat1 = patch([-0.5*batwidth,-0.5*batwidth,0.5*batwidth,0.5*batwidth], ...
	     [-1,-1+opt.batheight,-1+opt.batheight,-1], ...
	     opt.bat_color);
set(bat1,'EdgeColor','none','EraseMode','xor')
bat2 = patch([-0.5*batwidth,-0.5*batwidth,0.5*batwidth,0.5*batwidth], ...
	     [1,1-opt.batheight,1-opt.batheight,1], ...
	     opt.bat_color);
set(bat2,'EdgeColor','none','EraseMode','xor')
count = text(0,0,'');
set(count,'FontUnits','normalized','FontSize',0.2,'HorizontalAlignment','center','VerticalAlignment','middle','Color',opt.text_color,'Visible','off');
%if ori 
%    set(count,'Rotation',90);
%end 
    
score1 = text(0.975, -0.4, '');
set(score1,'FontUnits','normalized','FontSize',0.1,...
	   'HorizontalAlignment','center','VerticalAlignment','top',...
	   'Color',opt.text_color,'Visible','off');

score2 = text(0.975, 0.4, '');
set(score2,'FontUnits','normalized','FontSize',0.1,...
	   'HorizontalAlignment','center','VerticalAlignment','top',...
	   'Color',opt.text_color,'Visible','off');

%if ori
%    set(score1,'Rotation',90);
%    set(score2,'Rotation',90);
%end

ball_x = 0;
ball_y = 0;

dia = 0.05;
if opt.ballissquare,
  diaf= sqrt(2);
  circle_x = pi/4:pi/2:pi*7/4;
else
  diaf= 1;
  circle_x = linspace(0,2*pi,32);
end
circle_y = sin(circle_x);
circle_x = cos(circle_x);
if ori==1
    circle_y = circle_y/opt.position(3)*opt.position(4);
else
    circle_x = circle_x/opt.position(3)*opt.position(4);
end
circ_x = 0.5*circle_x*dia*diaf;
circ_y = 0.5*circle_y*dia*diaf;
    
ball = patch(circ_x,circ_y,[0.5,0.5,0.5],'EraseMode','xor');
set(ball,'EdgeColor','none', 'FaceColor',opt.ball_color);

drawnow;

blold = [];

if ischar(opt.save)
  movi = avifile(opt.save, 'fps',fs, ...
    'compression',opt.compression,'quality',opt.quality);
end

fwd= 1;
nFrames= 0;
writtenFrames= 0;
movie_nr= 1;
tt_movie_start= tt_start;
while run & ~isempty(out)
  % draw feedback.
  if iscell(out) & length(out)==10
    var1 = out{6};
    var2 = out{10};
    tt  = out{8}*1000;
    if tt>stop
      run = false;
      break;
    end
    if tt>=tt_old,
      fwd= 0;
    end
    while tt>tt_old+1000/fs
      waitForSync(1000/fs*(1-opt.speedup));
      if ischar(opt.save),
        if (tt-tt_movie_start)/1000>opt.max_length,
          movi= close(movi);
          fprintf('movie part %d: written %d frames.\n', movie_nr, ...
            nFrames-writtenFrames);
          writtenFrames= nFrames;
          if movie_nr==1,
            if isunix,
              unix(sprintf('mv %s.avi %s_1.avi', opt.save));
            else
              cmd= sprintf('rename %s.avi %s_1.avi', opt.save, opt.save);
              dos(cmd);
            end
          end
          movie_nr= movie_nr+1;
          movi = avifile([opt.save '_' int2str(movie_nr)], 'fps',fs, ...
            'compression',opt.compression, 'quality',opt.quality);
          tt_movie_start= tt;
        end
        nFrames= nFrames+1;
        F = getframe(gca, [1 1 opt.position(3:4)]);
        movi = addframe(movi,F);
      else
        drawnow
      end
%      fprintf('%g,%g\n',tt,tt_old);
      tt_old = tt_old+1000/fs;
    end
    
    array = [var1(1),var2];
    a = [0,array,zeros(1,5-length(array))];
    
    switch a(2)
     case POSITION
      set(gcf,'Position',a(3:end));
     case PLAYER1
      set(bat1,'XData',a(3)*(1-0.5*batwidth1)+[-0.5*batwidth1,-0.5*batwidth1,0.5*batwidth1,0.5*batwidth1]);
      old_pos1 = a(3);
      
     case PLAYER2
      set(bat2,'XData',-a(3)*(1-0.5*batwidth2)+[-0.5*batwidth2,-0.5*batwidth2,0.5*batwidth2,0.5*batwidth2]);
      old_pos2 = a(3);
      
     case VIEW 
%      ori = a(3);
      if opt.ballissquare,
        circle_x = pi/4:pi/2:pi*7/4;
      else
        circle_x = linspace(0,2*pi,32);
      end
      circle_y = sin(circle_x);
      circle_x = cos(circle_x);
      if ori
        circle_y = circle_y/opt.position(3)*opt.position(4);
        set(gca,'View',[-90 90]);
%        set([score1 score2 count],'Rotation',90);
      else
        circle_x = circle_x/opt.position(3)*opt.position(4);
        set(gca,'View',[0 90]);
%        set([score1 score2 count],'Rotation',0);
      end    
      circ_x = 0.5*circle_x*dia*diaf;
      circ_y = 0.5*circle_y*dia*diaf;
      
     case START 
      %nothing???
      
     case STOP
      set(count,'Visible','on','String','stopped');
      
     case PAUSE
      set(count,'Visible','on','String','paused');
      
     case BACKGROUND_COLOR
%      set(gcf,'Color',a(3:end-1));
      
     case BAT_COLOR1
%      set(bat1,'FaceColor',a(3:end-1));
      
     case BAT_COLOR2
%      set(bat2,'FaceColor',a(3:end-1));
      
     case BAT_WIDTH1 
      batwidth1 = a(3);
      set(bat1,'XData',old_pos1*(1-0.5*batwidth1)+[-0.5*batwidth1,-0.5*batwidth1,0.5*batwidth1,0.5*batwidth1]);
      
      
     case BAT_WIDTH2 
      batwidth2 = a(3);
      set(bat2,'XData',-old_pos2*(1-0.5*batwidth2)+[-0.5*batwidth2,-0.5*batwidth2,0.5*batwidth2,0.5*batwidth2]);
      
      
     case BAT_HEIGHT1
%      set(bat1,'YData',[-1,-1+a(3),-1+a(3),-1]);
%      batheight1 = a(3);
      
     case BAT_HEIGHT2
%      set(bat2,'YData',[1,1-a(3),1-a(3),1]);
%      batheight2 = a(3);
      
     case COUNTER_ON 
      if a(3)
        set(count,'Visible','on','String','');
      else
        set(count,'Visible','off');
      end
      
     case COUNTER 
      set(count,'String',int2str(a(3)));
      
     case BALL
      dx= sign(a(3)-ball_x);
      dy= sign(a(4)-ball_y);
      ball_x = a(3);
      ball_y = a(4);
      try
        set(ball,'XData',circ_x+ball_x);
        set(ball,'YData',circ_y+ball_y);
      end
      if ~fwd,
        if moveh~=0 & moveh==-dx,
%          fprintf('tick\n');
          if ~fwd & ~isempty(opt.border_audio),
            if opt.sound,
              wavplay(opt.border_audio, audio.fs, 'async');
            end
            ae= ae+1;
            audio.event(ae).time= tt-audio_offset;
            audio.event(ae).file= opt.border_sound;
          end
        end
        if movev~=0 & movev==-dy,
%          fprintf('tock\n');
          if movev==1,
            if ~isempty(opt.bat1_audio),
              if opt.sound,
                wavplay(opt.bat1_audio, audio.fs, 'async');
              end
              ae= ae+1;
              audio.event(ae).time= tt-audio_offset;
              audio.event(ae).file= opt.bat1_sound;
            end
          else
            if ~isempty(opt.bat2_audio),
              if opt.sound,
                wavplay(opt.bat2_audio, audio.fs, 'async');
              end
              ae= ae+1;
              audio.event(ae).time= tt-audio_offset;
              audio.event(ae).file= opt.bat2_sound;
            end
          end
        end
      end
      moveh= dx;
      movev= dy;
      
     case BALL_ON
      if a(3)
        set(ball,'Visible','on');
      else
        set(ball,'Visible','off');
      end
      
     case DIAM 
      dia = a(3);
      circ_x = 0.5*circle_x*dia*diaf;
      circ_y = 0.5*circle_y*dia*diaf;
      set(ball,'XData',circ_x+ball_x);
      set(ball,'YData',circ_y+ball_y);
      
      
     case BALL_COLOR
       if isequal(a(3:end-1),[0 0 1]),
         set(ball,'FaceColor',opt.ball_color);
       elseif isequal(a(3:end-1),[1 0 0]),
         set(ball,'FaceColor',opt.ballout_color);
%         fprintf('dööö\n');
         if ~fwd & ~isempty(opt.ballout_audio),
           if opt.sound,
             wavplay(opt.ballout_audio, audio.fs, 'async');
           end
           ae= ae+1;
           audio.event(ae).time= tt-audio_offset;
           audio.event(ae).file= opt.ballout_sound;
         end
       else
         set(ball,'FaceColor',a(3:end-1));
       end
       
       
     case SCORE_ON
      if a(3)
        set(score1,'Visible','on');
        set(score2,'Visible','on');
      else
        set(score1,'Visible','off');
        set(score2,'Visible','off');
      end
      
     case SCORE1
      set(score1,'String',sprintf('% 2d',a(3)));
      
     case SCORE2
      set(score2,'String',sprintf('% 2d',a(3)));
      
     case WINNER
      if a(3)
%        set(count,'String','Player 2 wins','Visible','on');
      else
%        set(count,'String','Player 1 wins','Visible','on');
      end
      
     case EXIT
      run = false;
    end
  end
  out = load_log;
end

load_log('exit');

if ischar(opt.save)
  fFrames= round(opt.freeze_end*fs);
  for kk= 1:fFrames,
    movi = addframe(movi,F);
  end
  movi = close(movi);
  if movie_nr>1,
    fprintf('movie part %d: written %d ', movie_nr, nFrames-writtenFrames);
  else
    fprintf('movie: written %d ', nFrames+fFrames);
  end
  if fFrames>0,
    fprintf('( + %d) ', fFrames);
  end
  fprintf('frames');
  if movie_nr>1,
    fprintf('.\nmovie: in total %d frames', nFrames+fFrames);
  end
  fprintf(' at %d Hz (%.3f s).\n', fs, (nFrames+fFrames)/fs);
  
  %% this shoud be general 'audio-struct -> wav file';
  file_list= unique({audio.event.file});
  soundwave= cell(1, length(file_list));
  for ii= 1:length(file_list),
    [soundwave{ii},fs_temp]= wavread(file_list{ii});
    if fs_temp~=audio.fs,
      error('mismatch in sampling rates');
    end
  end
  full_length= ceil((nFrames+fFrames)/fs*audio.fs);
  wave= zeros(full_length, 1);
  for ae= 1:length(audio.event),
    pos= round(audio.event(ae).time/1000*audio.fs);
    sno= strmatch(audio.event(ae).file, file_list, 'exact');
    slen= size(soundwave{sno},1);
    maxlen= min(slen, full_length-pos+1);
    iv_src= 1:maxlen;
    iv_trg= pos:pos+maxlen-1;
    wave(iv_trg)= soundwave{sno}(iv_src);
  end
  wavwrite(wave, audio.fs, opt.save);
  fprintf('audio: written %d samples at %d Hz (%.3f s).\n', full_length, audio.fs, ...
    full_length/audio.fs);
end
