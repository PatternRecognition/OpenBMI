function printprogress(typ,base);

persistent h ty hh
if nargin==1
    ty = typ;
    switch typ
        case 1
            hh = figure;
            h = uicontrol('Style','text','FontSize',20,'units','normalized','position',[0.1 0.1 0.8 0.8]);        
    end
elseif nargin==0
    close(hh);
else
    
    switch ty
        case 1
            set(h,'String',sprintf('%i von %i',typ,base));
    end
end


drawnow
            