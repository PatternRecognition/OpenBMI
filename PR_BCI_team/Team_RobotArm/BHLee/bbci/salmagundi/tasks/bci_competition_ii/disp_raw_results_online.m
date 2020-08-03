%function disp_raw_results_online

%% benjamin
data_dir= [DATA_DIR 'eegImport/bci_competition_ii/albany/'];
%% gilles
% data_dir= 'labels/';

list1={'AA','BB','CC'};

list={};

B={[] [] [] []};

nob=1;

for j=1:3
  for i=7:10
    list{nob}=[list1{j} sprintf('%03d',i)];
    nob=nob+1;
  end;
end;

for ss=1:length(list1)

  range=(1:4)+(ss-1)*4;
  B={[] [] [] []};

  for i=range
    if (i>=9), offset=0; else offset=1; end       %This is a tweak to align trialnrs...
    
    load([list{i} 'RESonline_raw']);
    
    ff=fopen([ data_dir list{i} 'LABELS.dat'],'r');
    A=fscanf(ff,'%f ',[4 inf]);
    fclose(ff);
    
    for j=1:4
      Iextr=find((A(3,:)==j));
      %    plot(sort(z(A(2,Iextr)+offset))); hold on
      B{j}=[B{j} z(A(2,Iextr)+offset)];
    end;
  end;

  figure;
  
  hold on;

  color={'b','r','g','k'};
  
  for i=1:4
    N(i,:)=histc(B{i},-40:40);
    plot(-40:40,N(i,:),color{i});
    P=B{i};
    C(i,1)=sum(P<cut(1));
    C(i,2)=sum(P>=cut(1) & P<cut(2));
    C(i,3)=sum(P>=cut(2) & P<cut(3));
    C(i,4)=sum(P>=cut(3));
  end
  
  C
  plot([cut; cut],repmat([0 30],[3 1])');

end;




