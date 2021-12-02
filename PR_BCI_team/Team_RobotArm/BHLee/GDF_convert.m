clear; clc;


%% sload
% biosig에 해당하는 폴더에서 biosig_installer.m을 실행시킨다.
% 아래 cd (.gdf 파일 경로)와 savepath를 실행시켜 경로를 저장해준다.
% filelist = {'motorexecution_subject11_run1.gdf', 'motorexecution_subject11_run2.gdf', 'motorexecution_subject11_run3.gdf', 'motorexecution_subject11_run4.gdf', 'motorexecution_subject11_run5.gdf', 'motorexecution_subject11_run6.gdf', 'motorexecution_subject11_run7.gdf', 'motorexecution_subject11_run8.gdf', 'motorexecution_subject11_run9.gdf', 'motorexecution_subject11_run10.gdf'};

cd D:\robotarm\ydData\horizon_data\subject1\MI/
savepath

datedir = dir('*.gdf');
filelist = {datedir.name};
[signal, H] = sload(filelist);

%% cnt

cnt.clab = transpose(H.Label);
cnt.fs = 250;
cnt.title = ['Converted .gdf file'];
cnt.file = ['MEdata'];
cnt.x = signal;

%% mrk

% Class name
% 1536 = 'elbow flexion'
% 1537 = 'elbsion'
% 1538 = 'supination'
% 1539 ow exten= 'pronation'
% 1540 = 'hand open'
% 1541 = 'hand close'
% 1542 = 'rest'
% mrk.className은 elbow flexion/extension, forearm supination/pronation,
% hand open/close (right upper limb)로 설정하였음.

classNum.elbow_flexion = 1536;
classNum.elbow_extension = 1537;
classNum.supination = 1538;
classNum.pronation = 1539;
classNum.hand_open = 1540;
classNum.hand_close = 1541;
classNum.rest = 1542;

mrk.pos = transpose(H.TRIG);
% 
L = H.EVENT.TYP;
L = L(L<1543 & L>1535);
L = transpose(L);

% for i = 1:420
%     A(i) = 4+7*(i-1);
% end

% for ii = 1: size(H.EVENT.TYP,1)
%     for jj = 1: size(L,2)
%         if ii == A(jj)
%             BB(jj) = H.EVENT.TYP(A(jj));
%         end
%     end
% end

mrk.toe = L;

mrk.fs = cnt.fs;
H.Classlabel = {'elbow_flexion', 'elbow_extension', 'forearm_supination', 'forearm_pronation', 'hand_open', 'hand_close', 'rest'};
mrk.className = H.Classlabel;
classNum = [1536; 1537; 1538; 1539; 1540; 1541; 1542];

mrk.y = zeros(7,size(mrk.toe,2))
for nn = 1:size(mrk.toe,2)
    switch mrk.toe(nn)
        case 1536
            mrk.y(1,nn) = 1;
        case 1537
            mrk.y(2,nn) = 1;
        case 1538
            mrk.y(3,nn) = 1;
        case 1539
            mrk.y(4,nn) = 1;
        case 1540
            mrk.y(5,nn) = 1;
        case 1541
            mrk.y(6,nn) = 1;
        case 1542
            mrk.y(7,nn) = 1;
    end
end


%% mnt
mnt.x = transpose(H.ELEC.Phi); 
mnt.y =transpose(H.ELEC.Theta);
mnt.pos_3d = transpose(H.ELEC.XYZ);
mnt.clab = transpose(H.Label);


save('subject1.mat', 'cnt', 'mrk', 'mnt');

