clear stat

stat = []
for i=1:50
    tic
    [myseq, bla] = pseudoRandMatrix(5,6,5,i,1,1);
    stat{i} = bla;
    stat{i}.time=toc
end

figure
for i=1:50
    plot(i,stat_1{i}.time,'b*') ; hold on ; grid on
end


for i=1:50
    plot(i,stat{i}.var_overallSeq,'r.') ; hold on ; grid on
end
 

% Teste verschiedene Stufen von Frame Kandidaten
stat_3_3 = []
for i=[1:2:30]
    tic
    [myseq, bla] = pseudoRandMatrix(5,6,5,15,1,1,i);
    stat_3_3{i} = bla;
    stat_3_3{i}.time=toc
end

stat_max = []
for i=[1:2:30]
    tic
    [myseq, bla] = pseudoRandMatrix(5,6,5,15,1,1,i);
    stat{i} = bla;
    stat{i}.time=toc
end




figure
for i=[1:2:30]
    %plot(i,max(max(stat_3_3{i}.count)),'b.') ; hold on ; grid on
    plot(i,length(find(stat_3_3{i}.count)>10),'r*') ; hold on ; grid on
end



figure
for i=1:5:30
    i
    hist(reshape(stat{i}.count, size(stat{i}.count,1)*size(stat{i}.count,2),[]),-1:20); grid on;
    xlim=[0 350];
    pause;
   
end


%% Reduce long Sequences to very short but excellent ones
Filename = 'Seq_Screensize_7x6_GroupSize_7_Frames_15';
load(Filename);

collect = [];
%figure
for i=[1:size(Sequences,2)]
    %plot(i,Sequences{i}.stat.var_overallSeq,'b.') ; hold on ; grid on
    collect = [collect Sequences{i}.stat.var_overallSeq];
    %plot(i,length(find(stat{i}.count)>9),'r*') ; hold on ; grid on
end
goodIdx=find (collect<prctile(collect,5));

Sequences(setdiff([1:size(Sequences,2)],goodIdx))=[];
save([Filename '_opt'],'Sequences')




%% collect = [];
%figure
for i=[1:1000]
    %plot(i,Sequences{i}.stat.var_overallSeq,'b.') ; hold on ; grid on
    collect = [collect Sequences{i}.stat.var_overallSeq];
    %plot(i,length(find(stat{i}.count)>9),'r*') ; hold on ; grid on
end
goodIdx=find (collect<prctile(collect,5));

Sequences(setdiff([1:size(Sequences,2)],goodIdx))=[];
save([Filename '_opt'],'Sequences')



for i=goodIdx
    %plot(i,Sequences{i}.stat.var_overallSeq,'b.') ; hold on ; grid on
    Sequences{i}.stat.var_overallSeq];
    %plot(i,length(find(stat{i}.count)>9),'r*') ; hold on ; grid on
end




figure
for i=1:5:30
    i
    hist(reshape(stat{i}.count, size(stat{i}.count,1)*size(stat{i}.count,2),[]),-1:20); grid on;
    xlim=[0 350];
    pause;
   
end
