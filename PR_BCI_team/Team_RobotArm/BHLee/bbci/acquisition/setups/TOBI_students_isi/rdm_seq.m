function [seq] = rdm_seq(parts,bool)
%Erzeugt eine zufälliger Sequenz mit den 9 Silben mit mindestens "parts"
%vielen Target-Silben
%   IN
%       parts   Anzahl der mindest Zieltöne
%       bool    Wenn gesetzt wird auf parts kein rand-Wert zuaddiert
%               0=false, sonst true
%   OUT
%       seq     zufällige Sequenz


pause on;
seq = [];

if ~bool
    parts = parts+randi(4)-1;     %verändert die Anzahl der Targets zufällig um 0 bis 3
end

for int=1:parts
    container1 = [3,5,6,7,8,9];
    container2 = [4 6 7 8 9];
    container3 = [1 4 5 7 8 9];
    container4 = [2 3 6 8 9];
    container5 = [1 3 7 9];
    container6 = [1 2 4 7 8];
    container7 = [1 2 3 5 6 9];
    container8 = [1 2 3 4 6];
    container9 = [1 2 3 4 5 7];
    
    numbers = [];
    numbers(1) = randi(9);
    if(int>1)
        while (seq((int-1)*9) == numbers(1) | seq((int-1)*9-1) == numbers(1) | seq((int-1)*9-2) == numbers(1))
            numbers(1) = randi(9);
        end
    end
    remove_neighbor(numbers(1));

    i=1;

    while(size(numbers,2)<9)
        zahl = randi(9);
        %  Eintrag schon in Array
        while(size(find(numbers == zahl),2)~=0)
            zahl = randi(9); 
        end
        if(int>1 & size(numbers,2) == 1)
            while (size(find(numbers == zahl))~=0 | seq((int-1)*9) == zahl | seq((int-1)*9-1) == zahl)
                zahl = randi(9); 
            end
        end
        if(int>1 & size(numbers,2) == 2)
            while (size(find(numbers == zahl))~=0 | seq((int-1)*9) == zahl)
                zahl = randi(9); 
            end
        end

        switch numbers(i)
            case 1
                if(size(container1,2) == 0 | (zahl~=2 & zahl~=4))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end  

            case 2
                if(size(container2,2) == 0 | (zahl~=1 & zahl~=5 & zahl~=3))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 

            case 3
                if(size(container3,2) == 0 | (zahl~=2 & zahl~=6))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
            case 4
                if(size(container4,2) == 0 | (zahl~=1 & zahl~=5 & zahl~=7))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
            case 5
                if(size(container5,2) == 0 | (zahl~=2 & zahl~=4 & zahl~=6 & zahl~=8))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
            case 6
                if(size(container6,2) == 0 | (zahl~=3 & zahl~=5 & zahl~=9))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
            case 7
                if(size(container7,2) == 0 | (zahl~=4 & zahl~=8))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
            case 8
                if(size(container8,2) == 0 | (zahl~=7 & zahl~=5 & zahl~=9))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
            case 9
                if(size(container9,2) == 0 | (zahl~=8 & zahl~=6))
                i=i+1;
                numbers(i)=zahl;
                remove_neighbor(zahl);
                end 
        end

    end
    seq = cat(2,seq,numbers);
end


    function remove_neighbor(zahl)
        
        container1 = container1(container1~=zahl);
        container2 = container2(container2~=zahl);
        container3 = container3(container3~=zahl);
        container4 = container4(container4~=zahl);
        container5 = container5(container5~=zahl);
        container6 = container6(container6~=zahl);
        container7 = container7(container7~=zahl);
        container8 = container8(container8~=zahl);
        container9 = container9(container9~=zahl);
    end
end