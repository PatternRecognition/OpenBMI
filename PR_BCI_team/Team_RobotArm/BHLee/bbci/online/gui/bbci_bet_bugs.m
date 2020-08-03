function bbci_bet_bugs(mode);
%BBCI_BET_BUGS starts a gui for bug-reports
%
% Guido Dornhege 04/05/05
% $Id: bbci_bet_bugs.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

persistent BBCI_BET_BUG BBCI_BET_BUG_EDIT BBCI_BET_BUG_CLEAN BBCI_BET_BUG_SEND BBCI_BET_BUG_SUBJECT

if nargin<1 | isempty(mode)
    mode = 1;
end


switch mode
 case 0
  % close the figure
  try
    close(BBCI_BET_BUG);
  end
 case 1
  % open the figure
  try
    close(BBCI_BET_BUG);
  end
  
  
  
  BBCI_BET_BUG = figure;
  sz = get(0,'ScreenSize');
  set(BBCI_BET_BUG,'MenuBar','none','Position',[sz(3)-405,sz(4)-405,400,400],'NumberTitle','off','Name','BUG_REPORT');
  
  [how,subj] = system('whoami');
  subj = subj(1:end-1);
  
  BBCI_BET_BUG_SUBJECT = uicontrol('Style','Edit','Units','normalized','Position',[0.35 0.9,0.59,0.08],'FontSize',12,'Tooltipstring','Type in your user id','String',subj);
  uicontrol('Style','Text','Units','normalized','position',[0.01 0.9 0.3 0.08],'FontSize',12,'Tooltipstring','who are you???','String','User Id');
  
  BBCI_BET_BUG_EDIT= uicontrol('Style','Edit','Units','normalized','Position',[0.01 0.1,0.98,0.78],'FontSize',12,'Tooltipstring','Type in your bug_message','String','','HorizontalAlignment','left','MAX',1000);
  
  BBCI_BET_BUG_CLEAN = uicontrol('Style','pushbutton','Units','normalized','Position',[0.01 0.01,0.4,0.08],'FontSize',12,'Tooltipstring','Clean the entry','String','Clean','Callback','bbci_bet_bugs(3);');

  BBCI_BET_BUG_SEND = uicontrol('Style','pushbutton','Units','normalized','Position',[0.59 0.01,0.4,0.08],'FontSize',12,'Tooltipstring','ADD the entry','String','SUBMIT BUG','Callback','bbci_bet_bugs(4);');

  
 case 3
  set(BBCI_BET_BUG_EDIT,'String','');
  [how,subj] = system('whoami');
  subj = subj(1:end-1);
  set(BBCI_BET_BUG_SUBJECT,'String',subj);
      
 case 4
  string = get(BBCI_BET_BUG_EDIT,'String');
  subject = get(BBCI_BET_BUG_SUBJECT,'String');
  set(BBCI_BET_BUG_EDIT,'String','Thanks for submitting the bug!!!');
  drawnow
  if ~isempty(string)
    bbci_bet_bugreport(subject,string);
  end
  set(BBCI_BET_BUG_EDIT,'String','');
  
end
      