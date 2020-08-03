function send_cnt(cnt, mrk, from, to)
% send_cnt - Send an eeg-File like the brainvision Recorder
%
% SYNOPSIS
%    
%    send_cnt(cnt)                     Send the full eeg data
%    send_cnt(cnt, <mrk , from, to>)   Send the eeg data with markers in a
%                                      specific range
%    send_cnt(file,<from, to>)         Load the eeg-Data from the file with 
%                                      loadProcessedEEG and send it.
%
% GLOBALS
%            WINDOW - (optional) Show the elapsed time in a figure
%                     (default: WINDOW = 0)
%            CYCLE  - (optional) Repeat when the end of the eeg-data is
%                     reached (default: CYCLE = 0)
%  
%
% ARGUMENTS
%                cnt  - The eeg-Data as a struct containing the fields
%                   .clab - The channel names
%                   .fs   - The sampling rate 
%                   .x    - The data points
%                mrk  - (optional) A struct containing the markers for the
%                       eeg-Data
%                   .desc - The description of the marker like 'S  1'
%                   .pos  - The position of the marker in the eeg-Data
%                   .fs   - Sampling rate of the markers (must be equal to
%                           cnt.fs)
%                from - (optional) The starting position in the eeg-Data
%                       (default: from = 1)
%                to    - (optional) The end position in the eeg-Data
%                        (default: to = nPoints)
%
%      
% RETURNS
%          nothing
%
% DESCRIPTION
%    A server is created and the program will wait till it gets the first
%    connection from a client. Thereafter it will send the eeg-Data in 40
%    ms intervals. When it reaches the end of the eeg-Data the connection
%    is closed.
%
% AUTHOR
%    ???
%    ????/??/?? - ???
%                   - file created
%    2008/05/16 - Max Sagebaum
%                   - documented and updated for the new maker format 
% (c) 2005 Fraunhofer FIRST

global WINDOW CYCLE

  % if window == 1 we will show the passed time in a figure
  if ~isempty(WINDOW) && WINDOW
    fig = figure;
    set(fig,'Position',[0 50 100 100]);
    h1 = uicontrol('Style','Text','Position',[5 55 80 40],'String', ...
       '0.0s','FontSize',18);  
    h2 = uicontrol('Style','Text','Position',[5 5 80 40],'FontSize',18);
    steps = 25;
    set(fig,'menubar','none');
    drawnow
  else 
    WINDOW = 0;
  end

  if isempty(CYCLE) 
    CYCLE = 0;
  end


  % if cnt is a char, we have the case that the function call was 
  % send_cnt(file,<from, to>)
  % so remap the arguments
  if ischar(cnt)
    if exist('from','var')
      to = from;
    end
    if exist('mrk','var')
      from = mrk;
    end
    [cnt,mrk] = loadProcessedEEG(cnt);
  end



  if ~exist('mrk','var') || isempty(mrk), 
    mrk= struct('pos',[], 'toe',[], 'fs',cnt.fs); 
  end
  if ~exist('from','var'), from=1; end
  if ~exist('to','var'), to=size(cnt.x,1); end
  if cnt.fs~=mrk.fs, error('mismatch of sampling rates'); end 

  send_cnt_like_bv(cnt, 'init');        % create a connection this method bocks until 
                                        % a client connects
  waitForSync(0);
  tt= 0.04*cnt.fs;                      % how many data points are in one data block
  pp= from;

  step = [];
  try
   while pp<=to-tt || CYCLE
     if pp>to-tt;
       pp = from;
     end

     inBlock = find(mrk.pos>=pp & mrk.pos<pp+tt); % the markers in the block
     if WINDOW
       set(h1,'String',sprintf('%7.1f s',pp/cnt.fs));
       if ~isempty(inBlock)
         curMarkers = mrk.desc(inBlock);
         curMarkersString = '';
         for markersPos = 1:size(curMarkers,2)
           curMarkersString = [curMarkersString curMarkers(markersPos)];
         end
           
         set(h2,'String',curMarkersString);
         step = steps;
       elseif ~isempty(step)
         step = step-1;
         if step==0
          set(h2,'String','');
          step = [];
         end

       end
       drawnow
     else
       fprintf('\r%7.1f s   ', pp/cnt.fs);
     end
     pause(0.040);

     pp= send_cnt_like_bv(cnt, pp, mrk.pos(inBlock), mrk.desc(inBlock));
   end
   fprintf('\r%40s\r','');

  catch
   fprintf('\nerror: %s\n', lasterr);

  end
  send_cnt_like_bv('close');
  if WINDOW
    close(fig);
  end
end
