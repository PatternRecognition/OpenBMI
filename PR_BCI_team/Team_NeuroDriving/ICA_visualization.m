%% ICA Visualization

cnt.x=fastica(cnt.x');
cnt.x=cnt.x';

fasticag(cnt.x');
% Call the graphical user interface of ICA

f=get(gcf,'child')
for i=1:34
    f(i).YLim=[-20 20];
end
% Revise the y scale of each channel