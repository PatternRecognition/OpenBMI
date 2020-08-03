function display_message(message)
im=ones(1080,1920,3);
% Make an image the same size and put text in it 
hf = figure('color','white','units','normalized','position',[.1 .1 .8 .8]);
image(im); 
set(gca,'units','pixels','position',[5 5 1920-1 1080-1],'visible','off')
text('units','pixels','position',[500 500],'fontsize',30,'string',message) 
tim = getframe(gca); 
close(hf) 
im=im*255;
im(1:size(tim.cdata,1),1:size(tim.cdata,2),:)=tim.cdata;
im=uint8(im);
fullscreen(im,2)
