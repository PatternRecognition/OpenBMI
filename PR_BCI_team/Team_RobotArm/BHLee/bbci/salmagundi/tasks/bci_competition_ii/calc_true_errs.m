function calc_true_errs

list1={'AA','BB','CC'};

list={};
nob=1;

for j=1:3
  for i=7:10
    list{nob}=[list1{j} sprintf('%03d',i)];
    nob=nob+1;
  end;
end;

for i=1:length(list)
  if (i>=9), offset=0; else offset=1; end       %This is a tweak to align trialnrs...

  load([list{i} 'RES']);

  ff=fopen([ 'labels/' list{i} 'LABELS.dat'],'r');
  A=fscanf(ff,'%f ',[4 inf]);
  fclose(ff);

  tot(i)=length(A(2,:));                        %total nb of trials
  errtot(i)=sum(A(3,:)~=predtargetpos(A(2,:)+offset));    %total nb of errors
  Iextr=find((A(3,:)==1) | A(3,:)==4);                    %indices for classes top and bottom
  totextr(i)=length(Iextr);                      
  orig(i)=sum(A(3,:)~=A(4,:));
  origextr(i)=sum(abs(A(3,Iextr)-A(4,Iextr))>1);          %errors for classes top and bottom
                                                          %(by grouping outputs 1-2 and 3-4)

  errextr(i)=sum(abs(A(3,Iextr)-predtargetpos(A(2,Iextr)+offset))>1);

  fprintf('%s Ours: %d ; Orig.: %d\n',list{i},sum(A(3,:)~=predtargetpos(A(2,:)+offset)), sum(A(3,:)~=A(4,:)));
end;


for i=1:length(list1)

range= (1:4)+(i-1)*4;

fprintf(1,'Subject %s: %f %% (%f %% for classes 1/4) ; Orig: %f %% / %f %%\n',...
	list1{i},...
	sum(errtot(range))/sum(tot(range))*100,...
	sum(errextr(range))/sum(totextr(range))*100,...
	sum(orig(range))/sum(tot(range))*100,...
	sum(origextr(range))/sum(totextr(range))*100);
end;

fprintf(1,'Total: %f %% (%f %% for classes 1/4) ; Orig: %f %% / %f %%\n',...
	sum(errtot)/sum(tot)*100,...
	sum(errextr)/sum(totextr)*100,...
	sum(orig)/sum(tot)*100,...
	sum(origextr)/sum(totextr)*100);



