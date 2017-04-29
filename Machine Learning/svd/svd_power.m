function[]=svd_power(data_file,M,iterations)
in_data=double(load(data_file));
A=in_data*transpose(in_data);
newA=in_data*transpose(in_data);
%--------------------'Matrix U:'------------------
u_mat=[];
for d=1:M  
   
   u_out=powermethod(A,iterations);
   u_mat(:,d)=u_out;
   %disp(u_out);
   for i=1:size(A,1)       
       val=transpose(u_out)*transpose(A(i,1:end))*u_out;
       A(i,1:end)=A(i,1:end)-transpose(val);
       
    end
end
fprintf('Matrix U:');
for i=1:size(u_mat,1)
    fprintf('\n Row   %d:',i);
    for j=1:size(u_mat,2)
        fprintf('%8.4f',u_mat(i,j));
    end
end
%----------------Matrix S-----------------------
%newA=in_data*transpose(in_data);
s_out=zeros(M,M);
for k=1:M
   for j=1:M
      if(k==j)
          s_out(k,j)=sqrt(transpose(u_mat(:,j))*newA*u_mat(:,j));
      end
   end
end
fprintf('\nMatrix S:');
for i=1:size(s_out,1)
    fprintf('\n Row   %d:',i);
    for j=1:size(s_out,2)
        fprintf('%8.4f',s_out(i,j));
    end
end

%--------------------'Matrix V:'------------------
B=transpose(in_data)*in_data;
v_mat=[];
for d=1:M  
   
   v_out=powermethod(B,iterations);
   v_mat(:,d)=v_out;
   %disp(u_out);
   for i=1:size(B,1)       
       val=transpose(v_out)*transpose(B(i,1:end))*v_out;
       B(i,1:end)=B(i,1:end)-transpose(val);
       
    end
end
fprintf('\nMatrix V:');
for i=1:size(v_mat,1)
    fprintf('\n Row   %d:',i);
    for j=1:size(v_mat,2)
        fprintf('%8.4f',v_mat(i,j));
    end
end
%------------------Reconstruction Matrix------------------
rec_mat=u_mat*s_out*transpose(v_mat);
fprintf('\n Reconstruction (U*S*V''):');
for i=1:size(rec_mat,1)
    fprintf('\n Row   %d:',i);
    for j=1:size(rec_mat,2)
        fprintf('%8.4f',rec_mat(i,j));
    end
end
fprintf('\n');
end