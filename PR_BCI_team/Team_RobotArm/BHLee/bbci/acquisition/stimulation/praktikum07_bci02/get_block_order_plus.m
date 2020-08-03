function [m_new]=get_block_order_plus(m)
%I.Sturm December 2007
%generates matrix with all combinations of keys plus probe tone. all sequences
%of a key are arranged in one block and three sequences in the same key with probe tones
%tonic third and fifth are added at the beginning of the block.

m_new=[];
column=0:12:size(m,2);
thirds=[4 3];
%permute the contents of one block
for i=1:size(column,2)-1
    pitch=m(1,column(i)+1);
    majmin=m(2,column(i)+1);
    temp=m(:,column(i)+1:column(i+1));
    
    temp=temp(:,randperm(12));
    prelude=[repmat(pitch,1,3);repmat(majmin,1,3);0 thirds(majmin) 7 ];
    temp=[prelude temp];
    m_new=[m_new temp];
   
end

%permute the blocks
order=randperm(24); 

%copy the old matrix
tmp_m=m_new;

%reorder the blocks according to 'order'
for i=1:size(order,2)
      % m(:,column(i)+1:column(i+1))=tmp_m(:,column(order(i))+ 1 :column(order(i)+1));
      
      m_new(:,15*(i-1)+1:15*i)=tmp_m(:,(order(i)-1)*15+1:(order(i)*15));
end



