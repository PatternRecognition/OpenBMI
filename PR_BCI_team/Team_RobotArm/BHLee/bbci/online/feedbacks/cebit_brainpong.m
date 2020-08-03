fig1 = figure;
set(fig1,'MenuBar','none','DoubleBuffer','on', 'BackingStore','off',...
        'Renderer','OpenGL','RendererMode','auto')
set(fig1,'Position',[200 -200 880 200]);

imagesc(randn(400))
axis off
set(gca,'Position',[0 0 1 1]);

fig2 = figure;
set(fig2,'MenuBar','none','DoubleBuffer','on', 'BackingStore','off',...
        'Renderer','OpenGL','RendererMode','auto')
set(fig2,'Position',[200 -1024 880 200]);

imagesc(randn(400))
axis off
set(gca,'Position',[0 0 1 1]);
