function [t_d] = normalize(data,m,s)
%NORMALIZE 
for i=1:size(data,1)
   for j=1:size(data,2)
      data(i,j)=(data(i,j)-m(j))/s(j); 
   end
end
%disp(data);
t_d=data;
end

