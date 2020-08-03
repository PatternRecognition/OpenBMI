function bbci_bet_message(str,varargin);
%BBCI_BET_MESSAGE CREATES A MESSAGE FOR BBCI_BET
% usage:
%   'create' bbci_bet_message(1);
%   'close'  bbci_bet_message(0);
%   'write'  bbci_bet_message(...), like fprintf without fid
%
% if bbci_bet_message called without created message_box string is 
% directly printed to standard out
% 
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 09/03/05
% $Id: bbci_bet_message.m,v 1.1 2006/04/27 14:24:59 neuro_cvs Exp $

persistent figerl tex te;

if isempty(figerl)
  figerl = 0;
end


if isnumeric(str)
  if str==0
    try
      close(figerl);
    end
    figerl = 0;
  else
    figerl = figure;
    scs = get(0,'ScreenSize');
    set(figerl,'Position',[0.5*(scs(3)-800),0.5*(scs(4)-400),800,400]);
    set(figerl,'MenuBar','none');
    set(figerl,'NumberTitle','off');
    set(figerl,'Name','Status: Wait');
    set(figerl,'Units','Pixel');
    set(figerl,'Color',[1 1 1]);
    axis off
    set(gca,'XLim',[0 800]);
    set(gca,'YLim',[0 400]);
    te = text(400,60,'');
    set(te,'HorizontalAlignment','center');
    set(te,'VerticalAlignment','bottom');
    set(te,'FontSize',20);
    tex = '';
  end
else
  if figerl==0
    fprintf(str,varargin{:});
  else
    tex = [tex,sprintf(str,varargin{:})];
    set(te,'String',sprintf('%s\n',tex));
    handlefigures('vis');
    drawnow;
  end
end
