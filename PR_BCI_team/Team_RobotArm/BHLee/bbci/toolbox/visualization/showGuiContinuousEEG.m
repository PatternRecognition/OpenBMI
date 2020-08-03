function showGuiContinuousEEG(mode,cnt,ival, mrk);

% showGuiContinuousEEG visualizes raw data, 
% time interval and channel selection can be set interactively
%
% mode = 1 for first display
%
% Christine Carl 23/10/06


persistent SHOW_EEG_FIG SHOW_EEG_INTERVAL SHOW_EEG_UPDATE CNT MRK SHOW_EEG_DURATION SHOW_EEG_ERR_FIG SHOW_EEG_ERROR_BUTTON SHOW_EEG_REJECT_CHANNEL SHOW_EEG_REJECT_BUTTON





switch mode
 case 0
  % close the figure
  try
    close(SHOW_EEG_FIG);
  end
 case 1
  % open the figure
  try
    close(SHOW_EEG_FIG);
  end
  
  
  CNT = cnt;
  MRK = mrk;
  
  SHOW_EEG_FIG = figure;
  sz = get(0,'ScreenSize');
  set(SHOW_EEG_FIG,'Position',[0,0,sz(3),sz(4)],'NumberTitle','off','Name','EEGDATA');
  
  h_axes = axes( 'Position',[0.1, 0.1, 0.65 ,0.8],...
    'Parent',SHOW_EEG_FIG);


  ival_str = num2str(ival);
  showContinuousEEG(CNT,ival, MRK);
  % default duration
  duration = ival(1,2) -ival(1,1);
  
  % for specifying a whole interval
  SHOW_EEG_INTERVAL = uicontrol('Style','Edit','Units','normalized','Position',[0.8 0.75,0.17,0.1],'FontSize',12,'Tooltipstring','Type in the interval you want to plot','String',ival_str);
  
  SHOW_EEG_UPDATE = uicontrol('Style','pushbutton','Units','normalized','Position',[0.8 0.60,0.17,0.1],'FontSize',12,'Tooltipstring','Update the plot','String','Update','Callback','showGuiContinuousEEG(2)');
  
  SHOW_EEG_INT_LABEL = uicontrol('Style','Text','Units','normalized','Position',[0.8 0.85,0.17,0.025],'FontSize',12,'String','Interval for visualization');
  
  
  
  % for "scrolling"
  SHOW_EEG_DURATION = uicontrol('Style','Edit','Units','normalized','Position',[0.8 0.25,0.17,0.05],'FontSize',12,'Tooltipstring','Type in the length of the interval for visualization','String',duration);
  
  SHOW_EEG_DUR_LABEL = uicontrol('Style','Text','Units','normalized','Position',[0.8 0.3,0.17,0.025],'FontSize',12,'String',' Scolling with specified interval duration');
  
  SHOW_EEG_RIGHT = uicontrol('Style','pushbutton','Units','normalized','Position',[0.9 0.1,0.08,0.1],'FontSize',12,'Tooltipstring','Show following interval','String','Forward','Callback','showGuiContinuousEEG(3)');
  
  SHOW_EEG_LEFT = uicontrol('Style','pushbutton','Units','normalized','Position',[0.8 0.1,0.08,0.1],'FontSize',12,'Tooltipstring','Show previous interval','String','Backward','Callback','showGuiContinuousEEG(4)');
  
  
  SHOW_EEG_REJECT_CHANNEL = uicontrol('Style','Edit','Units','normalized','Position',[0.8 0.4,0.08,0.025],'String', 'Channels to reject');
  
  SHOW_EEG_REJECT_BUTTON = uicontrol('Style','pushbutton','Units','normalized','Position',[0.9 0.4,0.08,0.025],'String','Reject','Callback', 'showGuiContinuousEEG(5)');
  
  
 case 2
  ival_str = get(SHOW_EEG_INTERVAL, 'String');
  ival = str2num(ival_str);
  if ival(1,2)<=ival(1,1)|ival(1,1)<0|ival(1,2)>(size(CNT.x,1)*1000/CNT.fs-100)|size(ival,2)~=2
  	err_win = error_handling(1);
	ival = [ (size(CNT.x,1)*1000/CNT.fs-5000) (size(CNT.x,1)*1000/CNT.fs-100)];
	waitfor(err_win);
	figure(SHOW_EEG_FIG);
 end
  
   showContinuousEEG(CNT,ival, MRK);
   set(SHOW_EEG_INTERVAL, 'String',num2str(ival)); 
   
  
 case 3
  duration = get(SHOW_EEG_DURATION, 'String');
  ival_str = get(SHOW_EEG_INTERVAL,'String');
  ival = str2num(ival_str);
  if (ival(1,2)+str2num(duration))>(size(CNT.x,1)*1000/CNT.fs)
	ival = [ival(1,2) (size(CNT.x,1)*1000/CNT.fs-100)]; 
	err_win = error_handling(1);
	waitfor(err_win);
	figure(SHOW_EEG_FIG);	
  else 
  	ival = [ival(1,2) ival(1,2)+str2num(duration)];
  end
  showContinuousEEG(CNT,ival, MRK);
  set(SHOW_EEG_INTERVAL, 'String',num2str(ival));   
   
   
 case 4
  duration = get(SHOW_EEG_DURATION, 'String');
  ival_str = get(SHOW_EEG_INTERVAL,'String');
  ival = str2num(ival_str);
  ival = [ max(0,ival(1,1)-str2num(duration)) ival(1,1)];
  showContinuousEEG(CNT,ival, MRK);
  set(SHOW_EEG_INTERVAL, 'String',num2str(ival));   
  


case 5

  channel = get(SHOW_EEG_REJECT_CHANNEL,'String');
  CNT = proc_selectChannels(CNT, 'not', channel);
  

end

return



function h = error_handling(mode)

persistent SHOW_EEG_ERR_FIG
	
	SHOW_EEG_ERR_FIG = figure;
	h = SHOW_EEG_ERR_FIG
	
	
  	SHOW_EEG_ERROR = uicontrol('Style','Text','Units','normalized','Position',[0.25 0.5,0.5,0.2],'FontSize',12,'String','Backward','String','The specified interval is not in the range of the recorded data. Data is shown until the end of recordings. Press ok to continue');
	
	SHOW_EEG_ERROR_BUTTON = uicontrol('Style','pushbutton','Units','normalized','Position',[0.8 0.1,0.08,0.1],'FontSize',12,'String','Ok','Callback','close(gcf)');
	
	
	
return


