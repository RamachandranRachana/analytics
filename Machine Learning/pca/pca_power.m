function[]=pca_power(training_file,test_file,M,iterations)
A=load(training_file);
B=load(test_file);
B=double(B(:,1:end-1));
T=double(A(:,end));
A=double(A(:,1:end-1));
%disp(A);
dimension=size(A,2);
%disp(A(:,2));
projection_mat=[];
for d=1:M
    %A=
    %sd=cov(A(:,d),1);
    sd=cov(A,1);
    %disp(sd);
    %ud=powermethod(sd,iterations,dimension);
    ud=powermethod(sd,iterations);
    fprintf('Eigenvector %3d\n',d);
    %disp(ud)
    for j=1:size(ud,1)
        fprintf('%3d: %.4f\n',j,ud(j));
    end
    projection_mat(:,d)=ud;
    %A=A-(ud*A*transpose(ud));
    for i=1:size(A,1)
       %A(i,d+1)=A(i,d)-(transpose(ud)*A(i,d)*(ud));
       val=transpose(ud)*transpose(A(i,1:end))*ud;
       A(i,1:end)=A(i,1:end)-transpose(val);
       %A(i)=(A(i)-(transpose(ud)*A(i)*ud));
       %disp(A(i));
    end
    %disp(A);
    
end
projection_mat=transpose(projection_mat);

for k=1:size(B,1)
    %disp(transpose(B(k,1:end)));
   test_obj=projection_mat*transpose(B(k,1:end)); 
   fprintf('Test object %3d\n',k);
   %disp(size(test_obj));
   for e=1:size(test_obj,1)
      fprintf('%3d: %.4f\n',e,test_obj(e)); 
   end
end
%disp(A);
end