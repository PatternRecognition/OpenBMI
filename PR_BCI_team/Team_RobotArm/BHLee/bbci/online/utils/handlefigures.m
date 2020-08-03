function out = handlefigures(typ,varargin);
%HANDLEFIGURES handle figures
%
% usage:
%  <use>      out = handlefigures('use',name,anz);
%  <visible>  handlefigures('vis',vis,<name>);
%  <changes>  handlefigures('changes',name,prop1,propvalue1,...);
%  <close>    handlefigures('close',<name>);
%  <get>      out = handlefigures('get');
%
% input:
%  name    name of the figure
%          if use and figure with this name exists, this figures will be activated
%  vis     on(true) or off(false) (name can be empty in this case 
%          and means all figures)
%
% output:
%  out     use: the figure number
%          get: fignames
%
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 08/03/2005
% $Id: handlefigures.m,v 1.5 2006/11/23 09:55:28 neuro_cvs Exp $

persistent figs fignames visflag active_fig

if isempty(visflag)
  visflag = 'on';
end

switch typ
 case 'use'
  if length(varargin)==1
    nfig = 1;
  else
    nfig = varargin{2};
  end
  ind = find(strcmp(varargin{1},fignames));
  if ~isempty(ind);
    out = figs{ind};
    if nfig==length(out)
      active_fig(ind) = 1;
      set(out(2:end),'Visible','off');
      set(out(1),'Visible','on');
      figure(out(1));
      return;
    else
      figspos = ind;
      for a = 1:length(ind)
        set(figs(ind(a)),'CloseRequestFcn','closereq;');
      end
      try
        delete([figs{ind}]);
      end
    end
  else
    figspos = length(figs)+1;
  end
  figs{figspos} = zeros(1,nfig);
  if nfig>1
    fprintf('\npress space in the activated figure to switch\n');
  end
  for i = 1:nfig
    f = figure;
    set(f,'MenuBar','none');
    set(f,'NumberTitle','off');
    set(f,'Name',varargin{1});
    set(f,'Visible',visflag);
    set(f,'CloseRequestFcn',['handlefigures(''close'',''' varargin{1} ''');']);
    if nfig>1
      set(f,'KeyPressFcn',sprintf('ch = get(gcbo,''CurrentCharacter''); if ch==32, handlefigures(''next_fig'',''%s'');end',varargin{1}));
    end
    figs{figspos}(i) = f;
  end
  fignames = cat(2,fignames,varargin(1));
  out = figs{figspos};
  active_fig(figspos) = 1;
  set(out(2:end),'Visible','off');
  set(out(1),'Visible','on');
  figure(out(1));
  
  resize_figures(figs);
  
 case 'next_fig'
  ind = find(strcmp(varargin{1},fignames));
  active_fig(ind) = active_fig(ind)+1;
  if active_fig(ind)>length(figs{ind})
    active_fig(ind) = 1;
  end
  set(figs{ind}, 'Visible','off');
  set(figs{ind}(active_fig(ind)), 'Visible','on');
  figure(figs{ind}(active_fig(ind)));
  
 case 'vis'
  if length(varargin)<=1 | isempty(varargin{2})
    ind = 1:length(figs);
  else
    ind =find(strcmp(varargin{2},fignames));
  end
  if length(varargin)<1 | isempty(varargin)
    flag = visflag;
  else
    flag = varargin{1};
  end
  if isnumeric(flag)
    if flag
      flag = 'on';
    else
      flag = 'off';
    end
  end 
  for i = 1:length(ind);
    set([figs{active_fig(ind(i))}],'Visible',flag);
  end
  visflag = flag;
 case 'changes'
  if isempty(varargin{1})
    ind = 1:length(figs);
  else
    ind =find(strcmp(varargin{1},fignames));
  end
  for i = 1:length(ind)
    set([figs{active_fig(ind(i))}],varargin{2:end});
  end
 case 'close'
  if length(varargin)==0| isempty(varargin{1})
    ind = 1:length(figs);
  else
    ind =find(strcmp(varargin{1},fignames));
  end
  set(figs{ind},'CloseRequestFcn','closereq;');
  try
    delete([figs{ind}]);
    figs(ind) = [];
    fignames(ind) = [];
    resize_figures(figs);
  end
 case 'get'
  out = fignames;
end
  


function resize_figures(figs);

scs = get(0,'ScreenSize');
if length(figs)>0
  
  co = ceil(sqrt(length(figs)));
  ro = ceil(length(figs)/co);
  hei = (scs(4)-20)/ro;
  wid = scs(3)/co;
  for i = 1:length(figs)
    set(figs{i},'Position',[mod(i-1,co)*wid,scs(4)-ceil(i/co)*hei,wid,hei-20]);
  end  
end
