function [mnt] = eegMontageHUT ;
% OUT  mnt       struct for electrode montage
%         .x     - x coordiante of electrode positions
%         .y     - y coordinate of electrode positions
%         .clab  - channel labels
%         .box_x - x coordinate of axes for grid view
%         .box_y - y coordinate of axes for grid view
%         .box_w - width of axes for grid view
%         .box_h - height of axes for grid view
%
% SEE  setDisplayMontage, calc_ext_10_10_pos

global MEG_RAW_DIR;

HUT_CLABS = {'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT9','FT7','FC5','FC1','FC2','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP3','CP1','CP2','CP4','TP8','TP10','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','PO4','PO8','O1','Oz','O2','Iz'};

posSystem = calc_pos_ext_10_10;
posSystem = setDisplayMontage(posSystem, 'large') ;


x= posSystem.x;
y= posSystem.y;
z= posSystem.z;
elab= posSystem.clab;

maz= max(z(:));
miz= min(z(:));
%ur= [0 0 miz-0.8*(maz-miz)];
ur= [0 0 -1.5];

la= (maz-ur(3)) ./ (z(:)-ur(3));
Ur= ones(length(z(:)),1)*ur;
pos2d= Ur + (la*ones(1,3)) .* ([x(:) y(:) z(:)] - Ur);
pos2d= pos2d(:, 1:2);
pos2d(z<0,:)= NaN;

nChans= length(HUT_CLABS);
mnt.x= NaN*ones(nChans,1);
mnt.y= NaN*ones(nChans,1);
mnt.pos_3d= NaN*ones(3, nChans);
mnt.box =  NaN*ones(2,nChans+1);
mnt.box_sz = NaN*ones(2,nChans+1) ;
for ei= 1:nChans,
  ii= chanind(elab, HUT_CLABS{ei});
  if ~isempty(ii),
    mnt.x(ei)= pos2d(ii, 1);
    mnt.y(ei)= pos2d(ii, 2);
    mnt.pos_3d(:,ei)= [x(ii) y(ii) y(ii)];
    mnt.box(:,ei) = posSystem.box(:,ii) ;
    mnt.box(:,ei) = posSystem.box_sz(:,ii) ;
  end
end
mnt.box(:,end) = posSystem.box(:,end) ;
mnt.box(:,end) = posSystem.box_sz(:,end) ;

radius= 1.9;
mnt.x= mnt.x/radius;
mnt.y= mnt.y/radius;
mnt.clab= HUT_CLABS;






