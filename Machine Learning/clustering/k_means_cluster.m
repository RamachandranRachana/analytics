function []=k_means_cluster(data_file,k,iterations)
A=load(data_file);
% removing the target
A=A(:,1:end-1);
% Assigning clusters
for i=1:size(A,1)
% Assigning fixed clusters
   %if double((mod(i,2)))==0
    %   c(i)=2;
   %else
       %c(i)=1;
   %end
 % Random clusters
 c(i)= datasample(1:k,1,'Replace',true);
end

c=transpose(c);
A=horzcat(A,c);
%Initialization mean
for j=1:k
        ind= A(:,end) == j; 
        B = A(ind,:);
        %disp(B);
        m(j,:)=mean(B);   
end
M=m(:,1:end-1);
%disp(M);
%tm=A(453,1:end-1);
%tm=vertcat(tm,A(276,1:end-1));
%disp(tm);
% Initialization error
e=computeError(A,M);
%disp(e);
fprintf('After initialization: error = %.4f\n',e);
for i=1:iterations    
    A=A(:,1:end-1);
% computing the distance
    for n=1:size(A,1)
        d=A(n,:)-M;  
        d=d.^2;   
        d=sum(d,2);
        %disp(d);
        d=sqrt(d);
        [m,id]=min(d);
        c(n)=id;
    end
    A=horzcat(A,c);
    
    m=[];
    for j=1:k
        ind= A(:,end) == j; 
        B = A(ind,:);
        %disp(mean(B));
        m(j,:)=mean(B);   
    end
    M=m(:,1:end-1);
    e=computeError(A,M);
    fprintf('After iteration %d: error = %.4f\n',i,e);
    %disp(e);
end
end