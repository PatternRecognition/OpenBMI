function music_presentation(varargin)
global VP_SCREEN VP_CODE BCI_DIR TODAY_DIR VLC_DIR SOUND_DIR STIM_DIR
% presents music
% before running music_presentation run setup_musical_tension 
% Synopsis:
% music_presentation(playlist)
% 
% Arguments:
% 
% 
% Triggers:
%   
% 251: beginning of relaxation period
% 252: beginning of main experiment (after countdown)
% 253: end of main experiment
% 254: end
% 1 for beginning of excerpt
% 2 for end of excerpt
% 
% 
% irene.sturm@mailbox.tu-berlin.de

%VLC_DIR='\Program Files (x86)\VideoLAN\VLC\vlc.exe';
STIM_DIR= [BCI_DIR 'acquisition\data\sound\muscog\musical_tension\mp3_als_wav\'];

if length(varargin)>0 & isnumeric(varargin{1}),
  opt= propertylist2struct(varargin{1}, varargin{2:end});
else
  opt= propertylist2struct(varargin{:});
end

opt= set_defaults(opt, ...
                  'joystick',0, ...
                  'break',8, ...
                  'test', 0, ...
                  'require_response', 1, ...
                  'background', 0.5*[1 1 1], ...
                  'countdown', 7, ...
                  'countdown_fontsize', 0.3, ...
                  'duration_intro', 7000, ...
                  'bv_host', 'localhost', ...
                  'msg_intro','Entspannen', ...
                  'msg_fin', 'Ende', ...
                  'mssg',[], ...
                  'show_image',1, ...
                  'image_file', [BCI_DIR 'acquisition/data/images/kob.jpg']);

playlist=textread([BCI_DIR 'acquisition/stimulation/musical_tension/' opt.playlist],'%s');
if isfield(opt,'delete_obj')
delete(opt.delete_obj(find(ishandle(opt.delete_obj))));
end
if ~ishandle(opt.handle_background),
  opt.handle_background= stimutil_initFigure(opt);
end
if opt.show_image
    [img,map] = imread(opt.image_file);
    image(img)
end
[h_msg, opt.handle_background]= stimutil_initMsg;
if opt.show_image
   set(h_msg, 'String',opt.msg_intro, 'Visible','on');image
drawnow;

end
drawnow;
waitForSync;


 if ~isempty(opt.bv_host),
  bvr_checkparport;
end


waitForSync;

if opt.test,
  fprintf('Warning: test option set true: EEG is not recorded!\n');
else
%   if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
%   else
%     warning('!*NOT* recording: opt.filename is empty');
%   end
  ppTrigger(251);
  waitForSync(opt.duration_intro);
end

if ~opt.show_image,
  pause(1);
  stimutil_countdown(opt.countdown, opt);
else
    waitForSync;
    waitForSync(opt.duration_intro);
    set(h_msg, 'String',opt.msg_intro, 'Visible','off');
end

ppTrigger(252);
%%%%%%%%%%%%%%%%%%beginning of main experiment
pause(1);
% tf=figure
% screen_size = get(0, 'ScreenSize')
% imshow('C:/users/irene/Desktop/kob.jpg','Border','tight');
% set(tf, 'Position', VP_SCREEN);

%how many excerpts?
if opt.test
        nr_of_excerpts=1;
else
        nr_of_excerpts=size(playlist,1);
end
%play music without ratings
if ~opt.joystick
    fprintf(1,'no ratings are recorded!\n')
    
        for i=1:nr_of_excerpts
        fn=playlist{i};
        try
            [y f]=wavread([STIM_DIR fn]);
        catch
            fprintf(['corrupt file' num2str(i) '\n'])
        end
        p = audioplayer(y, f); 
        ppTrigger(1);%trigger for begin of excerpt
        %start playing
        play(p,f)
        while(isplaying(p))
            pause(0.02)
        end
        ppTrigger(2);%trigger for end
        pause(opt.break)
        clear y
    end
else
    %play music and record ratings
    
    for i=1:nr_of_excerpts
        fn=playlist{i};
        try
            [y f]=wavread([STIM_DIR fn]);
        catch
            fprintf(['corrupt file' num2str(i) '\n'])
        end
        p = audioplayer(y, f); 
        %open log file for recording joystick data
        log_fid=open_joystick_log(fn);
        ppTrigger(1);%trigger for begin of excerpt
        % log message
        j_val=jst;
        fprintf(log_fid,'Time= %s Joystick_value= %f Marker= %i \n',datestr(now, 'HH:MM:SS.FFF'),j_val(2),1);
        %start playing
        play(p,f);
        %dos(['"' vlc_path '" "' audiofiles{i} '" &']);
        while(isplaying(p))
            j_val=jst;
            fprintf(log_fid,'Time= %s Joystick_value= %f Marker= %i \n',datestr(now, 'HH:MM:SS.FFF'),j_val(2),0);  
            pause(0.02)
        end
        j_val=jst;
        ppTrigger(2);%trigger for endtrigger for begin of excerpt of excerpt
        fprintf(log_fid,'Time= %s | Joystick_value= %f | Marker= %i \n',datestr(now, 'HH:MM:SS.FFF'),j_val(2),2);
        fclose(log_fid);
        clear j_val y f
        pause(opt.break)
    end
end

%close

set(h_msg, 'String',opt.msg_fin);
set(h_msg, 'Visible','on');


ppTrigger(254);
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(5);
delete(h_msg);

