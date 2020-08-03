function [m]=get_block_order(m)

m_old=m;
column=0:12:size(m,2);

%permute the contents of one block
for i=1:size(column,2) -1
    temp=m(:,column(i)+1:column(i+1));
    m(:,column(i)+1:column(i+1))=temp(:,randperm(12));
   
end

%permute the blocks
order=randperm(24); 

%copy the old matrix
tmp_m=m;

%reorder the blocks according to 'order'
for i=1:size(order,2)
       m(:,column(i)+1:column(i+1))=tmp_m(:,column(order(i))+ 1 :column(order(i)+1));
end



