function [hdr, dat, marker]=Convert_Emotiv(Temp_hdr,dat)

%% Channel mask in emotiv

Emotiv_chMask=[3:16]; 
Marker_Row=36;
hdr=Temp_hdr;

%% Calcultate data time
nD=length(dat);
mark=dat(Marker_Row,:);
dat=dat(Emotiv_chMask,:);
nC=length(Emotiv_chMask);

%% Making the mark infromation
nT=1;
for i=1:nD
    if mark(i)>0
        nT=nT+1;
        marker.mark.Latency(nT)=i;
        marker.mark.Type(nT)=mark(i);
        marker.mark.urevnt(nT)=nT;
    end
   
end
%% Making XYZ information
% Temp_hdr.label
% leadidcodexyz(hdr);
hdr.NS=Marker_Row;
hdr.Label=Temp_hdr.label;
Temp=leadidcodexyz(hdr);

%% Removing unrelated information
hdr.Origin_label=Temp_hdr.label;

hdr.chLoc.XYZ=Temp.ELEC.XYZ(Emotiv_chMask,:);
hdr.chLoc.Phi=Temp.ELEC.Phi(Emotiv_chMask);
hdr.chLoc.Theta=Temp.ELEC.Theta(Emotiv_chMask);
hdr.label=Temp_hdr.label(Emotiv_chMask);

% hdr.chLoc=Temp.ELEC;







% marker.location.XYZ=repmat(NaN,nC,3);
% 
% 
% 
% marker.location=load('Emotiv_location.mat');




%
% num1=length(hdr.label)
% num2=length(data)
% EEG.data.x=data';


% for i=1:length(num1)
%
%     if length(EEG.hdr.label{i})<4
%
%         Temp_Label=EEG.hdr.label(i);
%
%         EEG.chanel(1,k)=Temp_Label;
%
%         E(k,:)=data(i,:)
%
%         k=k+1;
%      end
%
% end



% EEG.hdr.label





%
% EEG.data.x=data(3:16,:)';
%
% EEG.data.fs=hdr.samples(1);
% EEG.data.nCh=14;
% EEG.data.chSet=hdr.label;
% EEG.data.chSet=hdr.label(1,3:16);

% Temp=data(36,:);
% % Num_data=
% Num_time=length(data(1,:));
%
% K=[];
% for i=1:Num_time
%
%     if Temp(i)>0
%         K=K+1;
%         EEG.data.Mark.Latency(K)=i;
%         EEG.data.Mark.Type(K)=Temp(i);
%         EEG.data.Mark.urevnt(K)=K;
%     end
%
%
%
% end













