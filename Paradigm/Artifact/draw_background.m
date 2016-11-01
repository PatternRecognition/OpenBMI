black=[0 0 0];white=[255 255 255];

Screen('FillRect', w, black, [wRect(3)/2-(s_width/2) wRect(1) wRect(3)/2+(s_width/2) wRect(1)+s_height]); % up
Screen('FillRect', w, black, [wRect(3)/2-(s_width/2) wRect(4)-s_height wRect(3)/2+(s_width/2) wRect(4)]); % down
Screen('FillRect', w, black, [wRect(1)  wRect(4)/2-(s_height/2)  wRect(1)+s_width   wRect(4)/2+(s_height/2)]); % left
Screen('FillRect',w,black,[wRect(3)-s_width wRect(4)/2-(s_height/2) wRect(3) wRect(4)/2+(s_height/2)]); % right
fixSize =80;
[X,Y] = RectCenter(wRect);
PixofFixationSize = 80;
FixCross = [X-1,Y-PixofFixationSize,X+1,Y+PixofFixationSize;X-PixofFixationSize,Y-1,X+PixofFixationSize,Y+1];
Screen('FillRect', w, black, FixCross');
