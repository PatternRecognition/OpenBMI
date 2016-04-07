function mark=Emotiv_Mark(data)
Mark_Mask=36;
Temp=data(36,:);
mark=Temp_data(Trigger_Mask,:)

Num_time=length(data(1,:));

for i=1:Num_time
    
    if mark(i)>0
        n=n+1;
        EEG.mark.Latency(n)=i;
        EEG.mark.Type(n)=mark(i);
        EEG.mark.urevnt(n)=n;
    end
    
    
end

%initiallize
% K=0;
% for i=1:Num_time
% 
%     if Temp(i)>0
%         K=K+1;
%         mark.Latency(K)=i;
%         mark.Type(K)=Temp(i);
%         mark.urevnt(K)=K;
%     end
%    
%    
% 
% end