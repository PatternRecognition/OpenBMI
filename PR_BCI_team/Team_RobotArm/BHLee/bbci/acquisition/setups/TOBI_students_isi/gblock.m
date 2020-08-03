function blocks = gblock(n)      

N = n*5;
blockbrut = zeros(N,8);
blocks = zeros(n,32);
p = zeros(N,1);
for i = 1:N;
    blockbrut(i,:) = randperm(8);
    idx = find(blockbrut(i,:)==1);
    if (idx~=8)
        p(i) = p(i)+(blockbrut(i,idx+1)-blockbrut(i,idx))>6;
    end
    idx = find(blockbrut(i,:)==2);
    if (idx~=8)
        p(i) = p(i)+(blockbrut(i,idx+1)-blockbrut(i,idx))>6;
    end
end

ok = (p==0);
blockisi = (blockbrut(ok==1,:));
blockisi = blockisi(1:n,:);
for i = 1:n,
for j = 1:8
    blocks(i,4*(j-1)+1) = blockisi(i,j);
    blocks(i,4*(j-1)+2) = blockisi(i,j);
    blocks(i,4*(j-1)+3) = blockisi(i,j);
    blocks(i,4*(j-1)+4) = blockisi(i,j);
end
end
return