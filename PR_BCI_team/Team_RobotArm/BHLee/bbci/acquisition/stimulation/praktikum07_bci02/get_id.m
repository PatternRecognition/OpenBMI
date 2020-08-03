function id=get_id(matrix,val)

for i=1:size(matrix,2)
    
    if matrix(:,i)== val
        
        id=i;
    end
    
    
end

