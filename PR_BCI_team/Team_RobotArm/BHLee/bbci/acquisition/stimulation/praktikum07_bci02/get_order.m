function order=get_order(m_old,m_new)

order=[];
for i=1:size(m_old,2)
     for j=1:size(m_new,2)
         no=find(m_old(:,i)==m_new(:,j));
          if size(no,1)==3
              order=[order,j];

          end
    end 
end


