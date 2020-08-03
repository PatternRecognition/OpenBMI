selection= {'FC5,1,2,6', 'C5,3,z,4,6', 'CP5,1,2,6', 'P3,z,4'};

all_clab= scalpChannels;
clab= all_clab(chanind(all_clab, selection));

mnt= getElectrodePositions(clab);
mnt.x= 1.2*mnt.x;
mnt.y= 1.2*mnt.y;
H= drawScalpOutline(mnt, 'showLabels',1);
printFigure('~/layout_picocap16', 'format','pdf', 'paperSize',[12 12]);
