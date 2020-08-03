function confusion_mat

list1={'AA','BB','CC'};

list={};
nob=1;

for j=1:3
  for i=7:10
    list{nob}=[list1{j} sprintf('%03d',i)];
    nob=nob+1;
  end;
end;

M=zeros(12,4,4);
O=M;

for i=1:length(list)
  if (i>=9), offset=0; else offset=1; end       %This is a tweak to align trialnrs...

  load([list{i} 'RES']);

  ff=fopen([ 'labels/' list{i} 'LABELS.dat'],'r');
  A=fscanf(ff,'%f ',[4 inf]);
  fclose(ff);

  tot(i)=length(A(2,:));                        %total nb of trials
  
  for j=1:tot(i)
    M(i,A(3,j),predtargetpos(A(2,j)+offset))=M(i,A(3,j),predtargetpos(A(2,j)+offset))+1;
    O(i,A(3,j),A(4,j))=O(i,A(3,j),A(4,j))+1;
  end;

end;


for i=1:length(list1)

range= (1:4)+(i-1)*4;

fprintf(1,'Subject %s:\n',list1{i});
squeeze(sum(M(range,:,:)))
squeeze(sum(O(range,:,:)))

end;

fprintf(1,'Total: \n');
squeeze(sum(M))
squeeze(sum(O))
